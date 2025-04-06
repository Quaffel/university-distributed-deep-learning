import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_features: typing.Mapping[str, typing.Literal["categorical", "numerical"]] = {
    "age": "numerical",
    "sex": "categorical",
    "cp": "categorical",
    "trestbps": "numerical",
    "chol": "numerical",
    "fbs": "categorical",
    "restecg": "categorical",
    "thalach": "numerical",
    "exang": "categorical",
    "oldpeak": "numerical",
    "slope": "categorical",
    "ca": "categorical",
    "thal": "categorical",
}

_targets: typing.Mapping[str, typing.Literal["categorical", "numerical"]] = {
    "target": "categorical",
}

_columns: typing.Mapping[str, typing.Literal["categorical", "numerical"]] = {
    **_features,
    **_targets,
}

feature_names = list(_features.keys())


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=np.float32)


class EncodingFunction(typing.Protocol):
    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame: ...


def encode_dataset(
    dataset: pd.DataFrame,
    *,
    categorical_encoder: EncodingFunction,
    numerical_encoder: EncodingFunction,
    encode_targets: bool = True,
) -> pd.DataFrame:
    categorical_features = [
        feature_name
        for feature_name, feature_type in _features.items()
        if feature_type == "categorical" and feature_name in dataset.columns
    ]

    numerical_features = [
        feature_name
        for feature_name, feature_type in _features.items()
        if feature_type == "numerical" and feature_name in dataset.columns
    ]

    encoded_categorical_dataset = categorical_encoder(dataset[categorical_features])
    encoded_numerical_dataset = numerical_encoder(dataset[numerical_features])

    return pd.concat([encoded_categorical_dataset, encoded_numerical_dataset], axis=1)


def as_tensor(dataset: pd.DataFrame, columns: list[str] | None = None) -> torch.Tensor:
    if columns is None:
        columns = list(dataset.columns)

    return torch.tensor(dataset[columns].values)


def partition_frame(
    frame: pd.DataFrame, split: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 <= split <= 1):
        raise ValueError(f"expect split to be a fraction of one, got {split}")

    pivot_element_idx = int(split * len(frame))

    return (
        frame.loc[:pivot_element_idx],
        frame.loc[pivot_element_idx:],
    )


def partition_series(series: pd.Series, split: float) -> tuple[pd.Series, pd.Series]:
    if not (0 <= split <= 1):
        raise ValueError(f"expect split to be a fraction of one, got {split}")

    pivot_element_idx = int(split * len(series))

    return (
        series.loc[:pivot_element_idx],
        series.loc[pivot_element_idx:],
    )


def build_client_datasets(
    dataset: pd.DataFrame,
    client_feature_name_mapping: list[list[str]],
    train_test_split: float,
    *,
    categorical_encoder: EncodingFunction,
    numerical_encoder: EncodingFunction,
    encode_targets: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    client_datasets: list[pd.DataFrame] = [
        encode_dataset(
            dataset[client_features],
            categorical_encoder=categorical_encoder,
            numerical_encoder=numerical_encoder,
            encode_targets=encode_targets,
        )
        for client_features in client_feature_name_mapping
    ]

    client_datasets_train, client_datasets_test = zip(
        *[
            partition_frame(client_dataset, train_test_split)
            for client_dataset in client_datasets
        ]
    )

    client_datasets_train = list(client_datasets_train)
    client_datasets_test = list(client_datasets_test)

    return (
        list(map(as_tensor, client_datasets_train)),
        list(map(as_tensor, client_datasets_test)),
    )


def build_target_dataset(
    dataset: pd.DataFrame,
    train_test_split: float,
    *,
    categorical_encoder: EncodingFunction,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = categorical_encoder(dataset["target"].to_frame())
    
    target_train, target_test = partition_frame(target, train_test_split)

    return as_tensor(target_train), as_tensor(target_test)
