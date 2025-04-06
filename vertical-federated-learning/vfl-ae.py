from torch import nn, optim
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from centralized import HeartDiseaseNN as EvaluatorModel

from components import preprocessing
from models.autoencoder import Autoencoder


class MseKldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


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


def _encode_categorical_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(dataset, columns=dataset.columns).astype("float32")


class NumericalFeatureEncoder:
    def __init__(self):
        self._scaler = StandardScaler()

    def train_and_transform_feature(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.fit_transform(dataset),
            columns=dataset.columns,
            index=dataset.index,
        )

    def transform_feature(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.transform(dataset),
            columns=dataset.columns,
            index=dataset.index,
        )


class CategoricalFeatureEncoder:
    def __init__(self):
        self._scaler = StandardScaler()

    def train_and_transform_feature(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = _encode_categorical_feature(dataset)
        return pd.DataFrame(
            self._scaler.fit_transform(dataset),
            columns=dataset.columns,
            index=dataset.index,
        )

    def transform_feature(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = _encode_categorical_feature(dataset)
        return pd.DataFrame(
            self._scaler.transform(dataset),
            columns=dataset.columns,
            index=dataset.index,
        )


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = preprocessing.load_dataset(
        Path(__file__).parent.parent / "datasets" / "heart" / "dataset.csv"
    )

    numerical_encoder = NumericalFeatureEncoder()
    categorical_encoder = CategoricalFeatureEncoder()

    dataset_features = dataset.drop("target", axis=1)
    dataset_targets = dataset["target"]

    def encode_features(dataset: pd.DataFrame, train: bool):
        if train:
            encode_categorical_feature = categorical_encoder.train_and_transform_feature
            encode_numerical_feature = numerical_encoder.train_and_transform_feature
        else:
            encode_categorical_feature = categorical_encoder.transform_feature
            encode_numerical_feature = numerical_encoder.transform_feature

        dataset = preprocessing.encode_dataset(
            dataset,
            categorical_encoder=encode_categorical_feature,
            numerical_encoder=encode_numerical_feature,
            encode_targets=False,
        )

        return preprocessing.as_tensor(dataset)

    dataset_features_train, dataset_features_test = preprocessing.partition_frame(dataset_features, 0.8)
    dataset_features_train = encode_features(dataset_features_train, True)
    dataset_features_test = encode_features(dataset_features_test, False)


    def encode_targets(feature: pd.Series):
        return torch.tensor(feature.values).long()

    dataset_target_train, dataset_target_test = map(
        encode_targets,
        preprocessing.partition_series(dataset_targets, 0.8),
    )

    model = Autoencoder(
        input_dimensions=dataset_features_train.shape[1] + 1,  # why +1?
        wide_hidden_dimensions=48,
        narrow_hidden_dimensions=32,
        latent_dimensions=16,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = MseKldLoss()

    real_data = torch.concat(
        (dataset_features_train, dataset_target_train.view(-1, 1)), dim=1
    )

    model.train_with_settings(
        epochs=200,
        batch_size=64,
        real_data=real_data,
        optimizer=optimizer,
        loss_function=loss_function,
    )

    _, mu, logvar = model.forward(real_data)

    synthetic_data = model.sample(len(real_data), mu.shape[1])
    synthetic_x = torch.tensor(synthetic_data[:, :-1])
    synthetic_y = torch.tensor(synthetic_data[:, -1]).long()

    print("--------------Testing model trained on real data----------")
    _run_evaluator_model(
        dataset_features_train,
        dataset_target_train,
        dataset_features_test,
        dataset_target_test,
    )

    print("--------------Testing model trained on synthetic data----------")
    _run_evaluator_model(
        synthetic_x, synthetic_y, dataset_features_test, dataset_target_test
    )


if __name__ == "__main__":
    main()
