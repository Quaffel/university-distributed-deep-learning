import typing

import numpy as np
import torch
from torch import nn


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
        narrow_hidden_dimensions=12,
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
                nn.Linear(wide_hidden_dimensions, narrow_hidden_dimensions),
                nn.BatchNorm1d(num_features=narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, narrow_hidden_dimensions),
                nn.BatchNorm1d(num_features=narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, latent_dimensions),
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
                nn.Linear(latent_dimensions, narrow_hidden_dimensions),
                nn.BatchNorm1d(narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, narrow_hidden_dimensions),
                nn.BatchNorm1d(num_features=narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, wide_hidden_dimensions),
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
