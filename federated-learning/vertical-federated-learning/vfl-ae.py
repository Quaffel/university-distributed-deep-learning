import itertools
from pathlib import Path
import typing

import numpy as np
import pandas as pd
import torch
from centralized import HeartDiseaseNN as EvaluatorModel
from components import encoders, partitions, preprocessing
from models.autoencoder import Autoencoder
from models.mlp import TwoLayerMlp
from sklearn.metrics import accuracy_score
from torch import nn, optim


class MseKldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


class AggregationModel(nn.Module):
    def __init__(
        self,
        client_encoders: list[TwoLayerMlp],
        client_decoders: list[TwoLayerMlp],
        auto_encoder: Autoencoder,
    ):
        super(AggregationModel, self).__init__()

        self.client_encoders = nn.ModuleList(client_encoders)
        self.client_decoders = nn.ModuleList(client_decoders)
        self.auto_encoder = auto_encoder

        # keep references to typed lists as well to access implementation-specific methods
        self._client_decoders = client_decoders
        self._client_encoders = client_encoders

    def decode(self, auto_encoder_output: torch.Tensor) -> torch.Tensor:
        decoder_inputs = torch.split(
            auto_encoder_output,
            [it.input_dimensions for it in self._client_decoders],
            dim=1,
        )

        decoder_outputs = [
            decoder_model(encoder_output)
            for decoder_model, encoder_output in zip(
                self.client_decoders, decoder_inputs
            )
        ]

        return torch.concat(decoder_outputs, dim=1)

    def forward(
        self, client_inputs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_outputs = [
            encoder_model(client_input)
            for encoder_model, client_input in zip(self.client_encoders, client_inputs)
        ]

        local_output = torch.concat(encoder_outputs, dim=1)
        auto_encoder_output, mean, log_variance = self.auto_encoder(local_output)

        return self.decode(auto_encoder_output), mean, log_variance

    def sample(self, samples: int, dimensions: int) -> torch.Tensor:
        with torch.no_grad():
            auto_encoder_output = self.auto_encoder.sample(samples, dimensions)
            return self.decode(auto_encoder_output)


def encode_features(
    dataset_train: pd.DataFrame, dataset_test: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    categorical_encoder = encoders.OneHotEncoder()
    numerical_encoder = encoders.MinMaxNumericalEncoder()

    dataset_train = preprocessing.encode_dataset(
        dataset_train,
        categorical_encoder=categorical_encoder.train_and_transform_feature,
        numerical_encoder=numerical_encoder.train_and_transform_feature,
    )

    dataset_test = preprocessing.encode_dataset(
        dataset_test,
        categorical_encoder=categorical_encoder.transform_feature,
        numerical_encoder=numerical_encoder.transform_feature,
    )

    return preprocessing.as_tensor(dataset_train), preprocessing.as_tensor(dataset_test)


def encode_target(
    dataset_train: pd.DataFrame, dataset_test: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        preprocessing.as_tensor(dataset_train).squeeze(dim=1).long(),
        preprocessing.as_tensor(dataset_test).squeeze(dim=1).long(),
    )


def prepare_dataset(
    dataset: pd.DataFrame,
    client_feature_name_mapping: list[list[str]],
    train_test_split: float,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    client_datasets_features_train, client_datasets_features_test = (
        preprocessing.build_client_datasets(
            dataset,
            client_feature_name_mapping,
            train_test_split,
        )
    )

    encoded_features_train, encoded_features_test = zip(
        *[
            encode_features(client_dataset_train, client_dataset_test)
            for client_dataset_train, client_dataset_test in zip(
                client_datasets_features_train, client_datasets_features_test
            )
        ]
    )

    dataset_target_train, dataset_target_test = preprocessing.build_target_dataset(
        dataset,
        train_test_split,
    )

    encoded_target_train, encoded_target_test = encode_target(
        dataset_target_train, dataset_target_test
    )

    return (
        list(encoded_features_train),
        encoded_target_train,
        list(encoded_features_test),
        encoded_target_test,
    )


def _build_model(
    *, client_datasets_features: list[torch.Tensor], latent_dimensions_per_input: int
) -> AggregationModel:
    client_encoders: list[TwoLayerMlp] = [
        TwoLayerMlp(
            client_dataset.shape[1],
            latent_dimensions_per_input * client_dataset.shape[1],
        )
        for client_dataset in client_datasets_features
    ]

    client_decoders: list[TwoLayerMlp] = [
        TwoLayerMlp(
            latent_dimensions_per_input * client_dataset.shape[1],
            client_dataset.shape[1],
        )
        for client_dataset in client_datasets_features
    ]

    auto_encoder = Autoencoder(
        input_dimensions=sum(it.output_dimensions for it in client_encoders),
        wide_hidden_dimensions=48,
        narrow_hidden_dimensions=32,
        latent_dimensions=16,
    )

    return AggregationModel(client_encoders, client_decoders, auto_encoder)


type Minibatch = list[torch.Tensor]


def _generate_minibatches(
    x_train: list[torch.Tensor],
    batch_size: int,
) -> typing.Iterable[Minibatch]:
    dataset_size = x_train[0].shape[0]

    start_idx = 0
    while start_idx + batch_size < dataset_size:
        end_idx = start_idx + batch_size

        yield [x[start_idx:end_idx] for x in x_train]
        start_idx = end_idx

    yield [x[start_idx:] for x in x_train]


def train_with_settings(
    model: nn.Module,
    epochs: int,
    batch_size: int,
    real_data: list[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_function,
):
    model.train()

    x_train = real_data
    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss = 0.0
        batches = 0
        for minibatch_data in _generate_minibatches(x_train, batch_size):
            batches += 1

            outs, mu, logvar = model.forward(minibatch_data)

            merged_minibatch_data = torch.concat(minibatch_data, dim=1)
            loss = loss_function(outs, merged_minibatch_data, mu, logvar)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {total_loss / batches:.3f}")


def merge(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.concat(tensors, dim=1)


def _run_evaluator_model(
    features_train: torch.Tensor,
    targets_train: torch.Tensor,
    features_test: torch.Tensor,
    targets_test: torch.Tensor,
):
    model = EvaluatorModel()
    optimizer = optim.AdamW(model.parameters())

    loss_function = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(1, 50):
        # train evaluator model
        optimizer.zero_grad()

        outputs_train = model(features_train)
        loss = loss_function(outputs_train, targets_train)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        predictions_train = torch.argmax(outputs_train, 1)
        train_accuracy = accuracy_score(targets_train, predictions_train)

        # test evaluator model
        outputs_test = model(features_test)

        predictions_test = torch.argmax(outputs_test, 1)
        test_accuracy = accuracy_score(targets_test, predictions_test)

        print(
            "Epoch {}, Loss: {:.2f}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(
                epoch, loss.item(), train_accuracy * 100, test_accuracy * 100
            )
        )


def _evaluate_autoencoder(
    *,
    model: AggregationModel,
    client_datasets_features_train: list[torch.Tensor],
    dataset_target_train: torch.Tensor,
    client_datasets_features_test: list[torch.Tensor],
    dataset_target_test: torch.Tensor,
):
    model.eval()
    _, mu, logvar = model.forward(client_datasets_features_train)

    synthetic_data = model.sample(len(dataset_target_train), mu.shape[1])
    synthetic_x = synthetic_data[:, :-1]

    synthetic_y = synthetic_data[:, -1]
    pred = synthetic_y.cpu().numpy()
    pred = np.clip(pred, 0, 1)
    pred = np.round(pred)
    synthetic_y = torch.tensor(pred).long()

    print("--------------Testing model trained on real data----------")
    _run_evaluator_model(
        merge(client_datasets_features_train)[:, :-1],
        dataset_target_train,
        merge(client_datasets_features_test)[:, :-1],
        dataset_target_test,
    )

    print("--------------Testing model trained on synthetic data----------")
    _run_evaluator_model(
        synthetic_x,
        synthetic_y,
        merge(client_datasets_features_test)[:, :-1],
        dataset_target_test,
    )


def main(
    *,
    clients: int = 4,
    latent_dimensions_per_input: int = 2,
    train_test_split: float = 0.8,
):
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = preprocessing.load_dataset(
        Path(__file__).parent.parent.parent / "datasets" / "heart" / "dataset.csv"
    )

    client_feature_name_mapping: list[list[str]] = (
        partitions.partition_elements_uniformly(
            [*preprocessing.feature_names, "target"], clients
        )
    )

    (
        client_datasets_features_train,
        dataset_target_train,
        client_datasets_features_test,
        dataset_target_test,
    ) = prepare_dataset(dataset, client_feature_name_mapping, train_test_split)

    model = _build_model(
        client_datasets_features=client_datasets_features_train,
        latent_dimensions_per_input=latent_dimensions_per_input,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = MseKldLoss()

    train_with_settings(
        model,
        epochs=200,
        batch_size=64,
        real_data=client_datasets_features_train,
        optimizer=optimizer,
        loss_function=loss_function,
    )

    _evaluate_autoencoder(
        model=model,
        client_datasets_features_train=client_datasets_features_train,
        dataset_target_train=dataset_target_train,
        client_datasets_features_test=client_datasets_features_test,
        dataset_target_test=dataset_target_test,
    )


if __name__ == "__main__":
    main()
