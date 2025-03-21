import time
import typing

import torch
from components.metrics import RunMetrics, RunResult
from components.server import DecentralizedServer
from components.tensor_types import IndexVector
from fedsgd.client import GradientClient
from torch.utils.data import Subset, DataLoader
from tqdm.notebook import tqdm


class FedSgdGradientServer(DecentralizedServer):
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        client_subsets: list[Subset],
        active_clients_fraction: float,
        learning_rate: float,
        seed: int,
    ) -> None:
        super().__init__(
            model=model,
            client_subsets=client_subsets,
            active_clients_fraction=active_clients_fraction,
            learning_rate=learning_rate,
            batch_size=-1,
            seed=seed,
            device=device,
        )
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(), lr=learning_rate
        )
        self.clients: list[GradientClient] = [
            GradientClient(device, model, subset) for subset in client_subsets
        ]
        self.client_datasets = client_subsets

    def select_clients(self) -> IndexVector:
        return self.generator.choice(len(self.clients), self.clients_per_round)

    def calculate_gradient_fraction_for_client(
        self,
        client: GradientClient,
        weights: list[torch.Tensor],
        seed: int,
        total_epoch_dataset_size: int,
    ) -> list[torch.Tensor]:
        client_dataset_size = client.client_data_size

        return [
            client_dataset_size / total_epoch_dataset_size * parameter_gradient
            for parameter_gradient in client.update(weights, seed)
        ]

    def run_epoch(self, weights: list[torch.Tensor], epoch: int) -> None:
        client_indices = [it.item() for it in self.select_clients()]
        client_dataset_size = sum(
            self.clients[client_idx].client_data_size for client_idx in client_indices
        )

        self.optimizer.zero_grad()

        # N x M; N clients with gradients for M parameters each
        client_gradients = [
            self.calculate_gradient_fraction_for_client(
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

        aggregated_client_gradients: list[torch.Tensor] = [
            # sum gradients parameter-wise; 'parameter_gradients' is a tuple that contains one gradient per client
            torch.stack(parameter_gradients, dim=0).sum(dim=0)
            for parameter_gradients in zip(*client_gradients)
        ]

        with torch.no_grad():
            for parameter, parameter_gradient in zip(
                self.model.parameters(), aggregated_client_gradients
            ):
                parameter.grad = parameter_gradient.to(self.device)

        self.model.train()
        self.optimizer.step()

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

        return RunResult("FedSgd", self.parameters, metrics)
