import typing

import torch
from components.client import AbstractClient
from torch.utils.data import Subset

# negative log likelihood loss
loss_function = torch.nn.functional.nll_loss


class WeightClient(AbstractClient):
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        client_data: Subset,
        learning_rate: float,
        batch_size: int,
        local_epochs: int,
    ) -> None:
        super().__init__(model, client_data, batch_size)
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(), lr=learning_rate
        )
        self.local_epochs = local_epochs
        self.client_data_size = len(client_data)
        self.device = device

    def train_epoch(
        self,
    ) -> None:
        self.model.train()

        for batch_features, batch_target in self.loader_train:
            batch_features = typing.cast(torch.Tensor, batch_features).to(self.device)
            batch_target = typing.cast(torch.Tensor, batch_target).to(self.device)

            self.optimizer.zero_grad()
            batch_output = self.model(batch_features)

            batch_loss = loss_function(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()

    def update(self, weights: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
        self.generator.manual_seed(seed)
        self.build_local_model(weights)

        for _ in range(self.local_epochs):
            self.train_epoch()

        parameter_weights = [
            parameter.detach().clone().cpu() for parameter in self.model.parameters()
        ]
        return parameter_weights
