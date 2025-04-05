import typing

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

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


def _encode_numerical_feature(feature: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        MinMaxScaler().fit_transform(pd.DataFrame(feature)),
        columns=[feature.name],
        index=feature.index,
    )


def _encode_categorical_feature(feature: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(pd.DataFrame(feature), columns=[feature.name]).astype(
        "float32"
    )


def _encode_feature(feature: pd.Series) -> pd.DataFrame:
    feature_name = feature.name
    if type(feature_name) != str:
        raise ValueError(
            f"name of feature series must be string, got {type(feature_name)}"
        )

    match _columns.get(feature_name, None):
        case "categorical":
            return _encode_categorical_feature(feature)
        case "numerical":
            return _encode_numerical_feature(feature)
        case None:
            raise ValueError(f"encountered unrecognized feature '{feature_name}'")


def _encode_dataset(dataset: pd.DataFrame, test: bool = False) -> pd.DataFrame:
    def preprocess_feature(feature_name: str) -> pd.DataFrame:
        if test and _columns.get(feature_name) == 'categorical':
            return pd.DataFrame(dataset[feature_name], columns=[feature_name], index=dataset.index)
        if not test and _columns.get(feature_name) == 'numerical':
            return pd.DataFrame(dataset[feature_name], columns=[feature_name], index=dataset.index)

        feature: pd.Series = dataset[feature_name]
        return _encode_feature(feature)

    encoded_client_features = [preprocess_feature(it) for it in dataset.columns]
    return pd.concat(encoded_client_features, axis=1)


def build_client_datasets(
    dataset: pd.DataFrame, client_feature_name_mapping: list[list[str]]
) -> typing.Tuple[list[torch.Tensor], torch.Tensor]:
    x_train = [
        torch.tensor(_encode_dataset(dataset[client_features]).values)
        for client_features in client_feature_name_mapping
    ]

    y_train = torch.tensor(_encode_dataset(dataset[["target"]]).values)

    return x_train, y_train
