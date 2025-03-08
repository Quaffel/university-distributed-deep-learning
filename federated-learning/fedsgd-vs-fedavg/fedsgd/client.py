import typing

import torch
from components.client import AbstractClient
from torch.utils.data import Subset

# negative log likelihood loss
loss_function = torch.nn.functional.nll_loss


class GradientClient(AbstractClient):
    def __init__(self, device: torch.device, model: torch.nn.Module, client_data: Subset) -> None:
        client_data_size = len(client_data)

        super().__init__(model, client_data, client_data_size)
        self.device = device
        self.client_data_size = client_data_size

    def update(self, weights: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
        self.generator.manual_seed(seed)
        client_model = self.build_local_model(weights)

        client_model.train()
        for batch_features, batch_targets in self.loader_train:
            batch_features = typing.cast(torch.Tensor, batch_features).to(self.device)
            batch_targets = typing.cast(torch.Tensor, batch_targets).to(self.device)

            batch_output = client_model(batch_features)

            batch_loss = loss_function(batch_output, batch_targets)
            batch_loss.backward()

        parameter_gradients = [
            typing.cast(torch.Tensor, param.grad).clone().detach().cpu()
            for param in client_model.parameters()
        ]
        return parameter_gradients
