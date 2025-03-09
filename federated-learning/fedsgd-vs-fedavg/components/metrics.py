import dataclasses

from pandas import DataFrame


@dataclasses.dataclass
class RoundParameters:
    clients_count: int
    active_clients_fraction: float
    batch_size: int
    local_epochs_count: int
    learning_rate: float
    seed: int


@dataclasses.dataclass
class RunMetrics:
    wall_time: list[float] = dataclasses.field(default_factory=list)
    message_count: list[int] = dataclasses.field(default_factory=list)
    test_accuracy: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RunResult:
    algorithm: str
    parameters: RoundParameters
    metrics: RunMetrics = dataclasses.field(default_factory=RunMetrics)

    def as_df(self) -> DataFrame:
        table_data = {
            "round": range(1, len(self.metrics.wall_time) + 1),
            "algorithm": self.algorithm,
            **dataclasses.asdict(self.parameters),
            **dataclasses.asdict(self.metrics),
        }

        df = DataFrame(table_data)
        df = df.rename(
            columns={
                "learning_rate": "\N{GREEK SMALL LETTER ETA}",
                "message_count": "message_count (sum)",
            }
        )

        return df