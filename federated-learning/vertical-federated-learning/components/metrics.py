import dataclasses


@dataclasses.dataclass
class EpochStatistics:
    mean_epoch_losses: list[float] = dataclasses.field(default_factory=list)
    mean_epoch_accuracies: list[float] = dataclasses.field(default_factory=list)
