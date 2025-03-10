import asyncio
import os
import sys
import typing
import math

import clients
import torch
import torch.distributed as dist
from torch.optim import Adam


def parse_cli_args() -> tuple[int, int, int]:
    arguments = sys.argv

    if len(arguments) != 4:
        print(
            f"syntax: <model rank> <stage rank> <stage size>, got {" ".join(arguments)}"
        )
        sys.exit(1)

    return int(arguments[1]), int(arguments[2]), int(arguments[3])


debug = bool(os.environ.get("DEBUG", False))
total_model_ranks = 3
model_rank, stage_rank, total_stage_ranks = parse_cli_args()


def calculate_client_rank(model_rank: int, stage_rank: int) -> int:
    return stage_rank * total_model_ranks + model_rank


rank = calculate_client_rank(model_rank=model_rank, stage_rank=stage_rank)
world_size = total_model_ranks * total_stage_ranks

if model_rank >= total_model_ranks:
    raise ValueError(
        f"expected model rank to be bounded by total model rank, got rank"
        f"{model_rank} / {total_model_ranks}"
    )

if stage_rank >= total_stage_ranks:
    raise ValueError(
        f"expected stage rank to be bounded by total stage rank, got rank"
        f"{stage_rank} / {total_stage_ranks}"
    )

print(
    f"running with rank {rank} "
    f"(model rank {model_rank} / {total_model_ranks} and stage rank {stage_rank} / {total_stage_ranks})"
)

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
    print(f"Using accelerator '{device}'")

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
else:
    device = torch.device("cpu")
    print("WARN: No accelerator found, running on CPU")

torch.manual_seed(0)


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)

# all groups, indexed by model rank
ranks_by_model_rank = [
    [
        stage_rank * total_model_ranks + model_rank
        for stage_rank in range(total_stage_ranks)
    ]
    for model_rank in range(total_model_ranks)
]
model_rank_groups = [dist.new_group(ranks=ranks) for ranks in ranks_by_model_rank]

micro_batch_size = 2
micro_batches = 2


def _build_client(model_rank: int, stage_rank: int) -> clients.RankClient:
    preceding_client = calculate_client_rank(
        model_rank=model_rank - 1, stage_rank=stage_rank
    )
    subsequent_client = calculate_client_rank(
        model_rank=model_rank + 1, stage_rank=stage_rank
    )

    if model_rank == 0:
        return clients.Rank0Client(
            device, subsequent_client, world_size, batch_size=micro_batch_size
        )
    elif model_rank == 1:
        return clients.Rank1Client(
            device,
            preceding_client,
            subsequent_client,
            world_size,
            batch_size=micro_batch_size,
        )
    elif model_rank == 2:
        return clients.Rank2Client(
            device, preceding_client, world_size, batch_size=micro_batch_size
        )
    else:
        raise ValueError(f"expected rank to be of value 0-2, got {model_rank}")


client = _build_client(model_rank, stage_rank)
optim = Adam(client.model.parameters(), lr=8e-4)


async def run_batch(batch: int):
    def launch_micro_batch(batch: int, micro_batch: int) -> typing.Awaitable[None]:
        micro_batch_id = batch * micro_batches + micro_batch

        return event_loop.create_task(
            client.run_mini_batch(micro_batch_id),
            name=f"micro batch {micro_batch_id} ({batch}-{micro_batch})",
        )

    micro_batch_tasks = [
        launch_micro_batch(batch, micro_batch) for micro_batch in range(micro_batches)
    ]
    await asyncio.gather(*micro_batch_tasks)


def get_gradients_from_model() -> list[torch.Tensor]:
    def get_parameter_gradient(parameter: torch.nn.Parameter) -> torch.Tensor:
        if parameter.grad is None:
            return torch.zeros_like(parameter)

        gradient = parameter.grad.detach().clone()
        return gradient

    return [
        get_parameter_gradient(parameter) for parameter in client.model.parameters()
    ]


def store_gradients_in_model(gradients: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, parameter_gradient in zip(client.model.parameters(), gradients):
            parameter.grad = parameter_gradient


def get_deflated_gradients() -> torch.Tensor:
    flattened_gradients = [
        parameter_gradient.view(-1) for parameter_gradient in get_gradients_from_model()
    ]

    return torch.concat(flattened_gradients)


def inflate_gradients(flattened_gradient: torch.Tensor) -> list[torch.Tensor]:
    parameter_shapes: list[tuple[int, ...]] = [
        parameter.shape for parameter in client.model.parameters()
    ]
    parameter_sizes: list[int] = [math.prod(shape) for shape in parameter_shapes]

    flattened_gradients = torch.split(flattened_gradient, parameter_sizes)

    return [
        parameter_flattened_gradient.view(parameter_shape)
        for parameter_flattened_gradient, parameter_shape in zip(
            flattened_gradients, parameter_shapes
        )
    ]


def as_asyncio_future(
    torch_task: torch.distributed.Work | None,
) -> asyncio.Future[None]:
    if torch_task is None:
        raise ValueError(
            "expected value other than None "
            "(was the operation destined at some process outside the current process group?)"
        )

    asyncio_future = asyncio.get_running_loop().create_future()

    asyncio_future.set_result(None)
    torch_task.wait()

    return asyncio_future


async def run_training():
    model_rank_group = model_rank_groups[model_rank]

    for batch in range(5_000):
        optim.zero_grad()

        await run_batch(batch)

        # take all gradients and reduce them into a single vector that can be used for aggregation
        deflated_gradients = get_deflated_gradients()

        # Sum up the gradients across all clients of the same model rank.
        # All clients with identical model rank form a pytorch.distributed group that can be used for aggregation.
        if debug:
            print(f"starting all reduce with group {ranks_by_model_rank[model_rank]}")

        await as_asyncio_future(
            dist.all_reduce(
                deflated_gradients,
                op=dist.ReduceOp.SUM,
                group=model_rank_group,
                async_op=True,
            )
        )

        # scale gradient by number of total stage ranks (for aggregation on mini batch-level; inter-client) and
        # number of micro batches (for aggregation on micro batch-level; intra-client)
        deflated_gradients.div(total_stage_ranks * micro_batches)

        gradients = inflate_gradients(deflated_gradients)
        store_gradients_in_model(gradients)

        optim.step()
        torch.cuda.empty_cache()


event_loop = asyncio.new_event_loop()

training_job = event_loop.create_task(run_training(), name="training")
event_loop.run_until_complete(training_job)
