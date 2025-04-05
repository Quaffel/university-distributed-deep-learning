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


class LossFunction(typing.Protocol):
    def __call__(
        self,
        outs: torch.Tensor,
        minibatch_data: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor: ...


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dimensions,
        wide_hidden_dimensions=50,
        dense_hidden_dimensions=12,
        latent_dimensions=3,
    ):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_dimensions, wide_hidden_dimensions),
                nn.BatchNorm1d(num_features=wide_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(wide_hidden_dimensions, dense_hidden_dimensions),
                nn.BatchNorm1d(num_features=dense_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(dense_hidden_dimensions, dense_hidden_dimensions),
                nn.BatchNorm1d(num_features=dense_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(dense_hidden_dimensions, latent_dimensions),
                nn.BatchNorm1d(num_features=latent_dimensions),
                nn.ReLU(),
            ),
        )

        self.latent_layer_mean = nn.Linear(latent_dimensions, latent_dimensions)
        self.latent_layer_log_variance = nn.Linear(latent_dimensions, latent_dimensions)

        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.Linear(latent_dimensions, latent_dimensions),
                nn.BatchNorm1d(latent_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(latent_dimensions, dense_hidden_dimensions),
                nn.BatchNorm1d(dense_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(dense_hidden_dimensions, dense_hidden_dimensions),
                nn.BatchNorm1d(num_features=dense_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(dense_hidden_dimensions, wide_hidden_dimensions),
                nn.BatchNorm1d(num_features=wide_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(wide_hidden_dimensions, input_dimensions),
                nn.BatchNorm1d(num_features=input_dimensions),
            ),
        )

    def encode(self, x):
        encoded = self.encoder(x)

        mean = self.latent_layer_mean(encoded)
        log_variance = self.latent_layer_log_variance(encoded)

        return mean, log_variance

    def reparameterize(self, mean, log_variance):
        if not self.training:
            # leave mean unchanged during inference
            return mean

        # create new samples based on the parameters predicted by the encoder
        std = log_variance.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, reparameterized_latent_representation: torch.Tensor):
        return self.decoder(reparameterized_latent_representation)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train_with_settings(
        self,
        epochs: int,
        batch_size: int,
        real_data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_function: LossFunction,
    ):
        num_batches = (
            len(real_data) // batch_size
            if len(real_data) % batch_size == 0
            else len(real_data) // batch_size + 1
        )
        for epoch in range(epochs):
            optimizer.zero_grad()

            total_loss = 0.0
            for minibatch in range(num_batches):
                if minibatch == num_batches - 1:
                    minibatch_data = real_data[int(minibatch * batch_size) :]
                else:
                    minibatch_data = real_data[
                        int(minibatch * batch_size) : int((minibatch + 1) * batch_size)
                    ]

                outs, mu, logvar = self.forward(minibatch_data)
                loss = loss_function(outs, minibatch_data, mu, logvar)
                total_loss += loss
                loss.backward()
                optimizer.step()

            print(
                f"Epoch: {epoch} Loss: {total_loss.detach().numpy() / num_batches:.3f}"
            )

    def sample(self, samples: int, dims) -> np.ndarray:
        # sigma = torch.exp(logvar / 2)
        sigma = torch.ones(dims)
        mu = torch.zeros(dims)

        q = torch.distributions.Normal(mu, sigma)
        z = q.rsample(sample_shape=torch.Size([samples]))
        with torch.no_grad():
            pred = self.decode(z).cpu().numpy()

        pred[:, -1] = np.clip(pred[:, -1], 0, 1)
        pred[:, -1] = np.round(pred[:, -1])
        return pred


class customLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    df = pd.read_csv(
        Path(__file__).parent.parent / "datasets" / "heart" / "dataset.csv"
    )
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    encoded_df = pd.get_dummies(df, columns=categorical)
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
    D_in = X.shape[1] + 1
    H = 48
    H2 = 32
    latent_dim = 16
    model = Autoencoder(D_in, H, H2, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mse_kld = customLoss()
    real_data = torch.concat((X_train, y_train.view(-1, 1)), dim=1)
    EPOCHS = 200
    BATCH_SIZE = 64
    model.train_with_settings(EPOCHS, BATCH_SIZE, real_data, optimizer, loss_mse_kld)

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
