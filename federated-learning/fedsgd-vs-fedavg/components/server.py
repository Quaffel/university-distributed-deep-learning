import typing
from abc import ABC

import numpy as np
import torch
from components.metrics import RoundParameters, RunResult
from torch.utils.data import DataLoader, Subset


class Server(typing.Protocol):
    def run(self, rounds: int) -> RunResult: ...


class AbstractServer(ABC, Server):
    def __init__(
        self, model: torch.nn.Module, parameters: RoundParameters, device: torch.device
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.device = device

        torch.manual_seed(parameters.seed)

    def evaluate_accuracy(self, test_loader: DataLoader) -> float:
        self.model.eval()

        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = typing.cast(torch.Tensor, batch_features).to(
                    self.device
                )
                batch_targets = typing.cast(torch.Tensor, batch_targets).to(self.device)

                batch_output: torch.Tensor = self.model(batch_features)

                # index of output neuron/logit corresponds to label
                batch_predictions = batch_output.argmax(dim=1, keepdim=True)

                correct_predictions += (
                    batch_predictions.eq(batch_targets.view_as(batch_predictions))
                    .sum()
                    .item()
                )
                total_predictions += batch_predictions.size(dim=0)

        print("correct: ", correct_predictions, "total: ", total_predictions)
        return correct_predictions / total_predictions


class DecentralizedServer(AbstractServer):
    def __init__(
        self,
        model: torch.nn.Module,
        client_subsets: list[Subset],
        active_clients_fraction: float,
        learning_rate: float,
        batch_size: int,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            model,
            RoundParameters(
                clients_count=len(client_subsets),
                active_clients_fraction=active_clients_fraction,
                batch_size=batch_size,
                local_epochs_count=1,
                learning_rate=learning_rate,
                seed=seed,
            ),
            device,
        )

        self.generator = np.random.default_rng(seed)
        self.clients_per_round = max(
            1, int(len(client_subsets) * active_clients_fraction)
        )
