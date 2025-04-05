from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from components import partitions, preprocessing
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


class BottomModel(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(BottomModel, self).__init__()
        self.local_out_dim = out_feat
        self.fc1 = nn.Linear(in_feat, out_feat)
        self.fc2 = nn.Linear(out_feat, out_feat)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.dropout(self.act(self.fc2(x)))


class TopModel(nn.Module):
    def __init__(self, local_models, n_outs):
        super(TopModel, self).__init__()
        self.in_size = sum(
            [local_models[i].local_out_dim for i in range(len(local_models))]
        )
        self.fc1 = nn.Linear(self.in_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        concat_outs = torch.cat(
            x, dim=1
        )  # concatenate local model outputs before forward pass
        x = self.act(self.fc1(concat_outs))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.dropout(x)


class AggregationModel(nn.Module):
    def __init__(
        self,
        local_models: list[nn.Module],
        output_dimension: int,
    ):
        super(AggregationModel, self).__init__()
        self._bottom_models = local_models
        self._top_model = TopModel(self._bottom_models, output_dimension)

        # TODO: Difference to Adam?
        self._optimizer = optim.AdamW(self.parameters())

        self._criterion = nn.CrossEntropyLoss()

    def train_with_settings(
        self,
        epochs: int,
        batch_size: int,
        client_datasets: list[torch.Tensor],
        dataset_targets: torch.Tensor,
    ):
        self.train()
        x_train = client_datasets
        y_train = dataset_targets

        dataset_size = y_train.shape[0]
        num_batches = (
            dataset_size // batch_size
            if dataset_size % batch_size == 0
            else dataset_size // batch_size + 1
        )

        for epoch in range(epochs):
            self._optimizer.zero_grad()
            total_loss = 0.0
            correct = 0.0
            total = 0.0

            for minibatch in range(num_batches):
                if minibatch == num_batches - 1:
                    x_minibatch = [x[int(minibatch * batch_size) :] for x in x_train]
                    y_minibatch = y_train[int(minibatch * batch_size) :]
                else:
                    x_minibatch = [
                        x[
                            int(minibatch * batch_size) : int(
                                (minibatch + 1) * batch_size
                            )
                        ]
                        for x in x_train
                    ]
                    y_minibatch = y_train[
                        int(minibatch * batch_size) : int((minibatch + 1) * batch_size)
                    ]

                outs = self.forward(x_minibatch)
                pred = torch.argmax(outs, dim=1)
                actual = torch.argmax(y_minibatch, dim=1)
                correct += torch.sum((pred == actual))
                total += len(actual)
                loss = self._criterion(outs, y_minibatch)
                total_loss += loss
                loss.backward()
                self._optimizer.step()

            print(
                f"Epoch: {epoch} Train accuracy: {correct * 100 / total:.2f}% Loss: {total_loss.detach().numpy()/num_batches:.3f}"
            )

    def forward(self, client_inputs: list[torch.Tensor]):
        local_outputs = [
            bottom_model(client_input)
            for bottom_model, client_input in zip(self._bottom_models, client_inputs)
        ]

        return self._top_model(local_outputs)

    def test(
        self,
        client_datasets: list[torch.Tensor],
        dataset_targets: torch.Tensor,
    ) -> tuple[float, float]:
        self.eval()
        x_test = client_datasets
        y_test = dataset_targets
        dataset_size = y_test.shape[0]

        with torch.no_grad():
            outs = self.forward(x_test)
            preds = torch.argmax(outs, dim=1)
            actual = torch.argmax(y_test, dim=1)
            accuracy = torch.sum((preds == actual)).div(dataset_size).item()
            loss = self._criterion(outs, y_test)
            return accuracy, loss


def build_client_datasets(
    dataset: pd.DataFrame,
    client_feature_name_mapping: list[list[str]],
    train_test_split: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    client_datasets = [
        preprocessing.encode_dataset(dataset[client_features])
        for client_features in client_feature_name_mapping
    ]

    client_datasets_train, client_datasets_test = zip(
        *[
            partitions.partition_frame(client_dataset, train_test_split)
            for client_dataset in client_datasets
        ]
    )

    client_datasets_train = list(client_datasets_train)
    client_datasets_test = list(client_datasets_test)

    return (
        list(map(preprocessing.as_tensor, client_datasets_train)),
        list(map(preprocessing.as_tensor, client_datasets_test)),
    )


def build_target_dataset(
    dataset: pd.DataFrame, train_test_split: float
) -> tuple[torch.Tensor, torch.Tensor]:
    target = preprocessing._encode_feature(dataset["target"])

    target_train, target_test = partitions.partition_frame(target, train_test_split)

    return preprocessing.as_tensor(target_train), preprocessing.as_tensor(target_test)


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

    client_datasets_train, client_datasets_test = build_client_datasets(
        dataset, client_feature_name_mapping, train_test_split
    )

    dataset_target_train, dataset_target_test = build_target_dataset(
        dataset, train_test_split
    )

    # model setup and training
    bottom_models: list[nn.Module] = [
        BottomModel(
            client_dataset.shape[1],
            client_output_dimensions_per_feature * client_dataset.shape[1],
        )
        for client_dataset in client_datasets_train
    ]

    model = AggregationModel(bottom_models, output_dimensions)

    model.train_with_settings(epochs, batch_size, client_datasets_train, dataset_target_train)

    # testing
    accuracy, loss = model.test(client_datasets_test, dataset_target_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%, test loss: {loss:.3f}")


if __name__ == "__main__":
    main()
