from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp import TwoLayerMlp
from components import partitions, preprocessing, encoders
from torch import nn
from tqdm import tqdm

# L_p normalization, maps every component to range from 0 to 1
# nn.functional.normalize()

# MinMaxScaler (makes significant difference as to how negative values are scaled)
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min


# import torch
#
# def scale_to_minus_one_to_one(tensor, min_val, max_val):
#     return 2 * (tensor - min_val) / (max_val - min_val) - 1
#
# # Example usage
# features = torch.tensor([[-5.0, 0.0, 10.0], [2.0, -3.0, 7.0]])
# min_vals = features.min(dim=0).values  # Minimum value for each feature
# max_vals = features.max(dim=0).values  # Maximum value for each feature
#
# normalized_features = scale_to_minus_one_to_one(features, min_vals, max_vals)
# print(normalized_features)


BottomModel = TwoLayerMlp


class TopModel(nn.Module):
    def __init__(self, local_models, n_outs):
        super(TopModel, self).__init__()

        input_dimensions = sum(it.output_dimensions for it in local_models)
        self.layer = nn.Sequential(
            nn.Sequential(nn.Linear(input_dimensions, 128), nn.LeakyReLU()),
            nn.Sequential(nn.Linear(128, 256), nn.LeakyReLU()),
            nn.Sequential(nn.Linear(256, 2), nn.LeakyReLU()),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # concatenate local model outputs before forward pass
        concat_outs = torch.concat(x, dim=1)

        return self.layer(concat_outs)


class AggregationModel(nn.Module):
    def __init__(
        self,
        local_models: list[nn.Module],
        output_dimension: int,
    ):
        super(AggregationModel, self).__init__()
        self._bottom_models = local_models
        self._top_model = TopModel(self._bottom_models, output_dimension)

    def forward(self, client_inputs: list[torch.Tensor]):
        local_outputs = [
            bottom_model(client_input)
            for bottom_model, client_input in zip(self._bottom_models, client_inputs)
        ]

        return self._top_model(local_outputs)


def test(
    *,
    model: nn.Module,
    client_datasets: list[torch.Tensor],
    dataset_targets: torch.Tensor,
    loss_function,
) -> tuple[float, float]:
    model.eval()
    x_test = client_datasets
    y_test = dataset_targets
    dataset_size = y_test.shape[0]

    with torch.no_grad():
        outs = model.forward(x_test)
        preds = torch.argmax(outs, dim=1)
        actual = torch.argmax(y_test, dim=1)
        accuracy = torch.sum((preds == actual)).div(dataset_size).item()
        loss = loss_function(outs, y_test)
        return accuracy, loss


def train_with_settings(
    *,
    model: nn.Module,
    epochs: int,
    batch_size: int,
    client_datasets: list[torch.Tensor],
    dataset_targets: torch.Tensor,
    loss_function,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    x_train = client_datasets
    y_train = dataset_targets

    dataset_size = y_train.shape[0]
    num_batches = (
        dataset_size // batch_size
        if dataset_size % batch_size == 0
        else dataset_size // batch_size + 1
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        correct = 0.0
        total = 0.0

        for minibatch in range(num_batches):
            if minibatch == num_batches - 1:
                x_minibatch = [x[int(minibatch * batch_size) :] for x in x_train]
                y_minibatch = y_train[int(minibatch * batch_size) :]
            else:
                x_minibatch = [
                    x[int(minibatch * batch_size) : int((minibatch + 1) * batch_size)]
                    for x in x_train
                ]
                y_minibatch = y_train[
                    int(minibatch * batch_size) : int((minibatch + 1) * batch_size)
                ]

            outs = model.forward(x_minibatch)
            pred = torch.argmax(outs, dim=1)
            actual = torch.argmax(y_minibatch, dim=1)
            correct += torch.sum((pred == actual))
            total += len(actual)
            loss = loss_function(outs, y_minibatch)
            total_loss += loss
            loss.backward()

            optimizer.step()

        print(
            f"Epoch: {epoch} Train accuracy: {correct * 100 / total:.2f}% Loss: {total_loss.detach().numpy()/num_batches:.3f}"
        )


def encode(
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


def prepare_dataset(
    dataset: pd.DataFrame,
    client_feature_name_mapping: list[list[str]],
    train_test_split: float,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    client_datasets_features_train, client_datasets_features_test = (
        preprocessing.build_client_datasets(
            dataset.drop("target", axis=1),
            client_feature_name_mapping,
            train_test_split,
        )
    )

    encoded_features_train, encoded_features_test = zip(
        *[
            encode(client_dataset_train, client_dataset_test)
            for client_dataset_train, client_dataset_test in zip(
                client_datasets_features_train, client_datasets_features_test
            )
        ]
    )

    dataset_target_train, dataset_target_test = preprocessing.build_target_dataset(
        dataset,
        train_test_split,
    )

    encoded_target_train, encoded_target_test = encode(dataset_target_train, dataset_target_test)

    return (
        list(encoded_features_train),
        encoded_target_train,
        list(encoded_features_test),
        encoded_target_test,
    )


def main(
    *,
    clients: int = 4,
    client_output_dimensions_per_feature=2,
    output_dimensions: int = 2,
    epochs: int = 300,
    batch_size: int = 64,
    train_test_split: float = 0.8,
):
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = preprocessing.load_dataset(
        Path(__file__).parent.parent / "datasets" / "heart" / "dataset.csv"
    )

    client_feature_name_mapping: list[list[str]] = (
        partitions.partition_elements_uniformly(
            list(preprocessing.feature_names), clients
        )
    )

    (
        client_datasets_features_train,
        dataset_target_train,
        client_datasets_features_test,
        dataset_target_test,
    ) = prepare_dataset(dataset, client_feature_name_mapping, train_test_split)

    # model setup and training
    bottom_models: list[nn.Module] = [
        BottomModel(
            client_dataset.shape[1],
            client_output_dimensions_per_feature * client_dataset.shape[1],
        )
        for client_dataset in client_datasets_features_train
    ]

    model = AggregationModel(bottom_models, output_dimensions)

    train_with_settings(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        client_datasets=client_datasets_features_train,
        dataset_targets=dataset_target_train,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(model.parameters()),
    )

    # testing
    accuracy, loss = test(
        model=model,
        client_datasets=client_datasets_features_test,
        dataset_targets=dataset_target_test,
        loss_function=nn.CrossEntropyLoss(),
    )

    print(f"Test accuracy: {accuracy * 100:.2f}%, test loss: {loss:.3f}")


if __name__ == "__main__":
    main()
