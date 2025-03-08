import time
import typing

import torch
from components.metrics import RunMetrics, RunResult
from components.server import DecentralizedServer
from components.tensor_types import IndexVector
from fedavg.client import WeightClient
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm


class FedAvgServer(DecentralizedServer):
    def __init__(
        self,
        device: torch.device,
        model_builder: typing.Callable[[], torch.nn.Module],
        learning_rate: float,
        batch_size: int,
        client_subsets: list[Subset],
        client_fraction: float,
        local_epochs: int,
        seed: int,
    ) -> None:
        super().__init__(
            model_builder(),
            client_subsets,
            client_fraction,
            learning_rate,
            batch_size,
            seed,
            device,
        )
        self.local_epochs_count = local_epochs
        self.clients = [
            WeightClient(
                device, model_builder(), subset, learning_rate, batch_size, local_epochs
            )
            for subset in client_subsets
        ]

    def select_clients(self) -> IndexVector:
        return self.generator.choice(len(self.clients), self.clients_per_round)

    def calculate_weight_fraction_for_client(
        self,
        client: WeightClient,
        weights: list[torch.Tensor],
        seed: int,
        total_epoch_dataset_size: int,
    ) -> list[torch.Tensor]:
        client_dataset_size = client.client_data_size

        return [
            client_dataset_size / total_epoch_dataset_size * parameter_weight
            for parameter_weight in client.update(weights, seed)
        ]

    def run_epoch(self, weights: list[torch.Tensor], epoch: int) -> None:
        client_indices = [it.item() for it in self.select_clients()]
        client_dataset_size = sum(
            self.clients[client_idx].client_data_size for client_idx in client_indices
        )

        # N x M; N clients with weights for M parameters each
        client_weights: list[list[torch.Tensor]] = [
            self.calculate_weight_fraction_for_client(
                self.clients[client_idx],
                weights,
                seed=self.parameters.seed
                + client_idx
                + 1
                + epoch * self.clients_per_round,
                total_epoch_dataset_size=client_dataset_size,
            )
            for client_idx in tqdm(client_indices, "clients", leave=False)
        ]

        aggregated_client_weights: list[torch.Tensor] = [
            # sum weights parameter-wise; 'parameter_weights' is a tuple that contains one weight vector per client
            torch.stack(parameter_weights, dim=0).sum(dim=0)
            for parameter_weights in zip(*client_weights)
        ]

        with torch.no_grad():
            for parameter, parameter_weight in zip(
                self.model.parameters(), aggregated_client_weights
            ):
                parameter[:] = parameter_weight.to(self.device)

    def run(self, rounds: int, test_loader: DataLoader) -> RunResult:
        metrics = RunMetrics()

        for epoch in tqdm(range(rounds), "epoch", leave=False):
            weights = [
                parameter.detach().clone() for parameter in self.model.parameters()
            ]

            wall_clock_start = time.perf_counter()
            weights = self.run_epoch(weights, epoch)
            wall_clock_end = time.perf_counter()

            accuracy = self.evaluate_accuracy(test_loader)
            execution_time_s = wall_clock_end - wall_clock_start

            metrics.test_accuracy.append(accuracy)
            metrics.wall_time.append(execution_time_s)
            metrics.message_count.append(2 * self.clients_per_round * (epoch + 1))

        return RunResult("FedAvg", self.parameters, metrics)
