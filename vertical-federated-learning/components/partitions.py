import typing

import pandas as pd


def partition_elements_uniformly(
    elements: list[str], partitions: int
) -> list[list[str]]:
    def generate_partitions():
        if partitions < 1:
            raise ValueError(f"expected at least one partition, got {partitions}")

        partition_size = len(elements) // partitions

        partition_start_idx = 0
        for _ in range(partitions - 1):
            partition_end_idx = partition_start_idx + partition_size
            yield elements[partition_start_idx:partition_end_idx]

            partition_start_idx = partition_end_idx

        yield elements[partition_start_idx:]

    return list(generate_partitions())


def partition_frame(
    frame: pd.DataFrame, split: float
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 <= split <= 1):
        raise ValueError(f"expect split to be a fraction of one, got {split}")

    pivot_element_idx = int(split * len(frame))

    return (
        frame.loc[:pivot_element_idx],
        frame.loc[pivot_element_idx:],
    )
