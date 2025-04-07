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

    def sample(self, samples: int, dims) -> torch.Tensor:
        # sigma = torch.exp(logvar / 2)
        sigma = torch.ones(dims)
        mu = torch.zeros(dims)

        q = torch.distributions.Normal(mu, sigma)
        z = q.rsample(sample_shape=torch.Size([samples]))
        with torch.no_grad():
            pred = self.decode(z).cpu().numpy()

        pred[:, -1] = np.clip(pred[:, -1], 0, 1)
        pred[:, -1] = np.round(pred[:, -1])

        return torch.tensor(pred)

    def sample_concrete(self, samples: int, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_variance / 2)

        q = torch.distributions.Normal(mean[0], sigma[0])
        z = q.rsample(sample_shape=torch.Size([samples]))
        with torch.no_grad():
            pred = self.decode(z).cpu().numpy()

        return torch.tensor(pred)
