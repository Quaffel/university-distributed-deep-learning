import typing
from torch import nn, optim
import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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


def _encode_numerical_feature(feature: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        StandardScaler().fit_transform(pd.DataFrame(feature)),
        columns=[feature.name],
        index=feature.index,
    )


def _encode_categorical_feature(feature: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(pd.DataFrame(feature), columns=[feature.name]).astype(
        "float32"
    )


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = preprocessing.load_dataset(
        Path(__file__).parent.parent / "datasets" / "heart" / "dataset.csv"
    )

    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    encoded_df = pd.get_dummies(dataset, columns=categorical)
    X = encoded_df.drop("target", axis=1)
    y = encoded_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train.values).long()
    y_test = torch.tensor(y_test.values).long()
    model = Autoencoder(
        input_dimensions=X.shape[1] + 1,  # why +1?
        wide_hidden_dimensions=48,
        narrow_hidden_dimensions=32,
        latent_dimensions=16,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = MseKldLoss()
    real_data = torch.concat((X_train, y_train.view(-1, 1)), dim=1)
    EPOCHS = 200
    BATCH_SIZE = 64
    model.train_with_settings(EPOCHS, BATCH_SIZE, real_data, optimizer, loss_function)

    _, mu, logvar = model.forward(real_data)

    synthetic_data = model.sample(len(real_data), mu.shape[1])
    synthetic_x = torch.tensor(synthetic_data[:, :-1])
    synthetic_y = torch.tensor(synthetic_data[:, -1]).long()

    print("--------------Testing model trained on real data----------")
    evalm1 = EvaluatorModel()
    opt1 = optim.AdamW(evalm1.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(1, 50):
        opt1.zero_grad()
        outputs = evalm1(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        opt1.step()
        losses.append(loss.item())
        _, preds_y = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train, preds_y)

        pred_test = evalm1(X_test)
        _, preds_test_y = torch.max(pred_test, 1)
        test_acc = accuracy_score(y_test, preds_test_y)
        print(
            "Epoch {}, Loss: {:.2f}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(
                epoch, loss.item(), train_acc * 100, test_acc * 100
            )
        )

    print("--------------Testing model trained on synthetic data----------")
    evalm2 = EvaluatorModel()
    opt2 = optim.AdamW(evalm2.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(1, 50):
        opt2.zero_grad()
        outputs = evalm2(synthetic_x)
        loss = criterion(outputs, synthetic_y)
        loss.backward()
        opt2.step()
        losses.append(loss.item())
        _, preds_y = torch.max(outputs, 1)
        train_acc = accuracy_score(synthetic_y, preds_y)

        pred_test = evalm2(X_test)
        _, preds_test_y = torch.max(pred_test, 1)
        test_acc = accuracy_score(y_test, preds_test_y)
        print(
            "Epoch {}, Loss: {:.2f}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(
                epoch, loss.item(), train_acc * 100, test_acc * 100
            )
        )
