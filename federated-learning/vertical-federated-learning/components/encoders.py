import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NormalNumericalEncoder:
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


class MinMaxNumericalEncoder:
    def __init__(self):
        self._scaler = MinMaxScaler()

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


class OneHotEncoder:
    def train_and_transform_feature(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return self.transform_feature(dataset)

    def transform_feature(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(dataset, columns=dataset.columns).astype("float32")
