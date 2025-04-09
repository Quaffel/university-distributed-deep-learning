from pathlib import Path
import typing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from components import encoders, partitions, preprocessing, metrics
from models.mlp import TwoLayerMlp
from torch import nn
from tqdm import tqdm

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
        local_models: nn.ModuleList,
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


def _evaluate_epoch(
    *,
    model: nn.Module,
    loss_function,
    client_datasets: list[torch.Tensor],
    dataset_targets: torch.Tensor,
) -> tuple[float, float]:
    model.eval()

    x_test = client_datasets
    y_test = dataset_targets
    dataset_size = y_test.shape[0]

    with torch.no_grad():
        outs = model.forward(x_test)
        preds = torch.argmax(outs, dim=1)
        accuracy = torch.sum((preds == y_test)).div(dataset_size).item()
        loss = loss_function(outs, y_test)

        return loss, accuracy


type Minibatch = tuple[list[torch.Tensor], torch.Tensor]


def _generate_minibatches(
    x_train: list[torch.Tensor],
    y_train: torch.Tensor,
    batch_size: int,
) -> typing.Iterable[Minibatch]:
    dataset_size = y_train.shape[0]

    start_idx = 0
    while start_idx + batch_size < dataset_size:
        end_idx = start_idx + batch_size
        x_minibatch = [x[start_idx:end_idx] for x in x_train]
        y_minibatch = y_train[start_idx:end_idx]

        yield x_minibatch, y_minibatch

        start_idx = end_idx

    x_minibatch = [x[start_idx:] for x in x_train]
    y_minibatch = y_train[start_idx:]

    yield x_minibatch, y_minibatch


def _train_epoch(
    *,
    model: nn.Module,
    batch_size: int,
    loss_function,
    optimizer: torch.optim.Optimizer,
    client_datasets_features: list[torch.Tensor],
    dataset_targets: torch.Tensor,
) -> tuple[float, float]:
    model.train()

    x_train = client_datasets_features
    y_train = dataset_targets

    optimizer.zero_grad()
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    batches = 0

    for x_minibatch, y_minibatch in _generate_minibatches(x_train, y_train, batch_size):
        batches += 1

        outs = model.forward(x_minibatch)
        pred = torch.argmax(outs, dim=1)
        correct += torch.sum((pred == y_minibatch)).item()
        total += len(y_minibatch)
        loss = loss_function(outs, y_minibatch)
        total_loss += loss.item()
        loss.backward()

        optimizer.step()

    epoch_mean_loss = total_loss / batches
    epoch_mean_accuracy = correct / total

    return epoch_mean_loss, epoch_mean_accuracy


def train(
    *,
    model: nn.Module,
    epochs: int,
    batch_size: int,
    loss_function,
    optimizer: torch.optim.Optimizer,
    client_datasets_features_train: list[torch.Tensor],
    dataset_targets_train: torch.Tensor,
    client_datasets_features_test: list[torch.Tensor],
    dataset_targets_test: torch.Tensor,
) -> tuple[metrics.EpochStatistics, metrics.EpochStatistics]:
    training_statistics = metrics.EpochStatistics()
    test_statistics = metrics.EpochStatistics()

    for _ in tqdm(range(epochs), "epoch", leave=False):
        epoch_loss_train, epoch_accuracy_train = _train_epoch(
            model=model,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer,
            client_datasets_features=client_datasets_features_train,
            dataset_targets=dataset_targets_train,
        )

        epoch_loss_test, epoch_accuracy_test = _evaluate_epoch(
            model=model,
            loss_function=loss_function,
            client_datasets=client_datasets_features_test,
            dataset_targets=dataset_targets_test,
        )

        training_statistics.mean_epoch_losses.append(epoch_loss_train)
        training_statistics.mean_epoch_accuracies.append(epoch_accuracy_train)

        test_statistics.mean_epoch_losses.append(epoch_loss_test)
        test_statistics.mean_epoch_accuracies.append(epoch_accuracy_test)

    return training_statistics, test_statistics


def _encode_features(
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


def _encode_target(
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
            dataset.drop("target", axis=1),
            client_feature_name_mapping,
            train_test_split,
        )
    )

    encoded_features_train, encoded_features_test = zip(
        *[
            _encode_features(client_dataset_train, client_dataset_test)
            for client_dataset_train, client_dataset_test in zip(
                client_datasets_features_train, client_datasets_features_test
            )
        ]
    )

    dataset_target_train, dataset_target_test = preprocessing.build_target_dataset(
        dataset,
        train_test_split,
    )

    encoded_target_train, encoded_target_test = _encode_target(
        dataset_target_train, dataset_target_test
    )

    return (
        list(encoded_features_train),
        encoded_target_train,
        list(encoded_features_test),
        encoded_target_test,
    )


def run(
    *,
    clients: int = 4,
    client_feature_name_mapping: list[list[str]] | None = None,
    client_output_dimensions_per_feature=2,
    output_dimensions: int = 2,
    epochs: int = 300,
    batch_size: int = 64,
    train_test_split: float = 0.8,
) -> tuple[metrics.EpochStatistics, metrics.EpochStatistics]:
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = preprocessing.load_dataset(
        Path(__file__).parent.parent.parent / "datasets" / "heart" / "dataset.csv"
    )

    if client_feature_name_mapping is None:
        client_feature_name_mapping = partitions.partition_elements_uniformly(
            list(preprocessing.feature_names), clients
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

    model = AggregationModel(nn.ModuleList(bottom_models), output_dimensions)

    return train(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(model.parameters()),
        client_datasets_features_train=client_datasets_features_train,
        dataset_targets_train=dataset_target_train,
        client_datasets_features_test=client_datasets_features_test,
        dataset_targets_test=dataset_target_test,
    )


if __name__ == "__main__":
    training_statistics, test_statistics = run()

    print("==== Training statistics ====")
    for accuracy, loss in zip(
        training_statistics.mean_epoch_accuracies, training_statistics.mean_epoch_losses
    ):
        print(f"{accuracy * 100:.2f}% loss: {loss:.3f}")

    print()
    print("==== Test statistics ====")
    for accuracy, loss in zip(
        test_statistics.mean_epoch_accuracies, test_statistics.mean_epoch_losses
    ):
        print(f"{accuracy * 100:.2f}% loss: {loss:.3f}")
