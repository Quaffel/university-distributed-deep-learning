import typing

import numpy as np
import numpy.random as npr
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from components.tensor_types import IndexVector


def _get_rng(generator_or_seed: npr.Generator | int) -> npr.Generator:
    if type(generator_or_seed) == int:
        return npr.default_rng(generator_or_seed)
    else:
        return typing.cast(npr.Generator, generator_or_seed)


def index_uniformly(
    dataset: datasets.MNIST,
    partitions_count: int,
    generator_or_seed: npr.Generator | int,
) -> list[IndexVector]:
    generator = _get_rng(generator_or_seed)

    shuffled_indices: IndexVector = generator.permutation(len(dataset))
    return np.array_split(shuffled_indices, partitions_count)


def _combine_partitions(
    mini_partitions: list[IndexVector],
    *,
    mini_partitions_per_partition: int,
    generator: npr.Generator,
) -> list[IndexVector]:
    if len(mini_partitions) % mini_partitions_per_partition != 0:
        raise ValueError(
            f"expected to have exactly {mini_partitions_per_partition} mini-partitions per partition,"
            f"got {len(mini_partitions)} mini-partitions"
        )

    partitions_count = len(mini_partitions) // mini_partitions_per_partition
    shuffled_partition_indices = generator.permutation(len(mini_partitions))

    return [
        np.concatenate(
            [mini_partitions[partition_idx] for partition_idx in mini_partition_indices]
        )
        for mini_partition_indices in shuffled_partition_indices.reshape(
            partitions_count, mini_partitions_per_partition
        )
    ]


def index_by_approximate_binary_target_partitions(
    dataset: datasets.MNIST,
    partitions_count: int,
    generator_or_seed: npr.Generator | int,
) -> list[IndexVector]:
    generator = _get_rng(generator_or_seed)

    targets = dataset.targets.clone().numpy()
    generator.shuffle(targets)

    sorted_indices: IndexVector = np.argsort(dataset.targets)
    sorted_indices_partitions: list[IndexVector] = np.array_split(
        sorted_indices, 2 * partitions_count
    )

    return _combine_partitions(
        sorted_indices_partitions, mini_partitions_per_partition=2, generator=generator
    )


def index_by_binary_target_partitions(
    dataset: datasets.MNIST,
    partitions_count: int,
    generator_or_seed: npr.Generator | int,
) -> list[IndexVector]:
    generator = _get_rng(generator_or_seed)

    targets = dataset.targets.clone().numpy()
    generator.shuffle(targets)

    client_indices = []
    unique_targets = np.unique(targets)

    unique_targets_count = unique_targets.shape[0]
    mini_partitions_per_label = partitions_count // unique_targets_count
    if partitions_count % unique_targets_count != 0:
        raise ValueError(
            "expected number of partitions to be a multiple of the number of unique Ïtargets, "
            f"got {partitions_count} partitions and {unique_targets_count} unique targets"
        )

    for target in unique_targets:
        label_indices = np.where(targets == target)[0]
        label_shards = np.array_split(label_indices, mini_partitions_per_label)
        client_indices.extend(label_shards)

    return _combine_partitions(
        client_indices, mini_partitions_per_partition=2, generator=generator
    )


def partition_dataset(
    dataset: Dataset, partitions: list[IndexVector]
) -> list[Subset[typing.Any]]:
    return [
        Subset(dataset, typing.cast(typing.Sequence[int], partition))
        for partition in partitions
    ]
