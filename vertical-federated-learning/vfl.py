from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler


from torch import nn

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
    def __init__(self, local_models, n_outs):
        super(AggregationModel, self).__init__()
        self.clients_count = None
        self.clients_features = None
        self.bottom_models = local_models
        self.top_model = TopModel(self.bottom_models, n_outs)

        # TODO: Difference to Adam?
        self.optimizer = optim.AdamW(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train_with_settings(
        self, epochs: int, batch_size: int, clients_count: int, clients_features, x, y
    ):
        self.clients_count = clients_count
        self.clients_features = clients_features
        x_train = [torch.tensor(x[features].values) for features in clients_features]
        y_train = torch.tensor(y.values)
        num_batches = (
            len(x) // batch_size
            if len(x) % batch_size == 0
            else len(x) // batch_size + 1
        )

        for epoch in range(epochs):
            self.optimizer.zero_grad()
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
                loss = self.criterion(outs, y_minibatch)
                total_loss += loss
                loss.backward()
                self.optimizer.step()

            print(
                f"Epoch: {epoch} Train accuracy: {correct * 100 / total:.2f}% Loss: {total_loss.detach().numpy()/num_batches:.3f}"
            )

    def forward(self, x):
        local_outs = [
            self.bottom_models[i](x[i]) for i in range(len(self.bottom_models))
        ]
        return self.top_model(local_outs)

    def test(self, x, y):
        x_test = [torch.tensor(x[feats].values) for feats in self.clients_features]
        y_test = torch.tensor(y.values)

        with torch.no_grad():
            outs = self.forward(x_test)
            preds = torch.argmax(outs, dim=1)
            actual = torch.argmax(y_test, dim=1)
            accuracy = torch.sum((preds == actual)) / len(actual)
            loss = self.criterion(outs, y_test)
            return accuracy, loss


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv("../datasets/heart/dataset.csv", dtype=np.float32)
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # scale numerical features for effective learning
    df[numerical_cols] = MinMaxScaler().fit_transform(df[numerical_cols])

    # convert categorical features to one-hot embeddings
    # returns boolean columns, convert them into float32 columns
    encoded_df = pd.get_dummies(df, columns=categorical_cols).astype("float32")
    num_clients = 4

    X = encoded_df.drop("target", axis=1)

    # TODO: How does selection of target columns work?
    # returns boolean columns, convert them into float32 columns
    Y = pd.get_dummies(encoded_df[["target"]], columns=["target"]).astype("float32")

    # "equally" partition the features
    feature_count_per_client: list[int] = (num_clients - 1) * [
        (len(df.columns) - 1) // num_clients
    ]
    feature_count_per_client.append(len(df.columns) - 1 - sum(feature_count_per_client))
    feature_count_per_client = np.array(feature_count_per_client)

    all_feature_names = list(df.columns)
    all_feature_names.pop()
    feature_names_per_client: list[list[str]] = []

    # sums up all elements in list; every element is mapped to sum up until that point
    encoded_df_feature_names = list(X.columns)
    start_index = 0
    for client_feature_count in feature_count_per_client:
        client_feature_names = all_feature_names[
            start_index : start_index + client_feature_count
        ]
        feature_names_per_client.append(client_feature_names)
        start_index = start_index + client_feature_count

    # insert names for columns created for one-hot encoding
    for i in range(len(feature_names_per_client)):
        updated_names = []
        for column_name in feature_names_per_client[i]:
            if column_name not in categorical_cols:
                # leave as-is
                updated_names.append(column_name)
                continue

            for name in encoded_df_feature_names:
                if "_" in name and column_name in name:
                    updated_names.append(name)

        feature_names_per_client[i] = updated_names

    # model architecture hyperparameters

    # does not have directly interpretable meaning ("latent space")
    client_outputs_per_feature = 2
    bottom_models = [
        # TODO: why are outputs multiplied by number of input features?
        BottomModel(
            len(input_dimension), client_outputs_per_feature * len(input_dimension)
        )
        for input_dimension in feature_names_per_client
    ]

    output_dimension = 2
    model = AggregationModel(bottom_models, output_dimension)

    # Training configurations
    EPOCHS = 300
    BATCH_SIZE = 64
    TRAIN_TEST_THRESH = 0.8

    # train-test-split
    X_train, X_test = (
        X.loc[: int(TRAIN_TEST_THRESH * len(X))],
        X.loc[int(TRAIN_TEST_THRESH * len(X)) + 1 :],
    )
    Y_train, Y_test = (
        Y.loc[: int(TRAIN_TEST_THRESH * len(Y))],
        Y.loc[int(TRAIN_TEST_THRESH * len(Y)) + 1 :],
    )

    model.train_with_settings(
        EPOCHS,
        BATCH_SIZE,
        num_clients,
        feature_names_per_client,
        X_train,
        Y_train,
    )

    accuracy, loss = model.test(X_test, Y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
