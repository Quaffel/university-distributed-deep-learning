import typing
from abc import ABC

import torch
from torch.utils.data import DataLoader, Subset


class Client(typing.Protocol):
    def update(self, weights: list[torch.Tensor], seed: int) -> list[torch.Tensor]: ...


class AbstractClient(ABC, Client):
    def __init__(
        self, model: torch.nn.Module, client_data: Subset, batch_size: int
    ) -> None:
        self.model = model
        self.generator = torch.Generator()
        self.loader_train = DataLoader(
            client_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=self.generator,
        )

    def build_local_model(
        self, weights: list[torch.Tensor]
    ) -> torch.nn.Module:
        with torch.no_grad():
            for client_parameter, server_parameter_values in zip(
                self.model.parameters(), weights
            ):
                client_parameter[:] = server_parameter_values
                client_parameter.grad = None

        return self.model
