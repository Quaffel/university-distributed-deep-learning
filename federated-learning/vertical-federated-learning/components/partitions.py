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
