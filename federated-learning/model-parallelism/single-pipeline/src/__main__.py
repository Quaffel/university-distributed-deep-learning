import asyncio
import os
from sys import argv
import typing

import clients
import torch
import torch.distributed as dist
from torch.optim import Adam

world_size = 3

rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"


if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
    print(f"Using accelerator '{device}'")

    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
else:
    device = torch.device("cpu")
    print("WARN: No accelerator found, running on CPU")


dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)

micro_batch_size = 1
micro_batches = 3


def _build_client(
    rank: int,
) -> clients.RankClient:
    if rank == 0:
        return clients.Rank0Client(device, world_size, batch_size=micro_batch_size)
    elif rank == 1:
        return clients.Rank1Client(device, world_size, batch_size=micro_batch_size)
    elif rank == 2:
        return clients.Rank2Client(device, world_size, batch_size=micro_batch_size)
    else:
        raise ValueError(f"expected rank to be of value 0-2, got {rank}")


client = _build_client(rank)
optim = Adam(client.model.parameters(), lr=8e-4)


async def run_training():
    event_loop = asyncio.get_running_loop()

    def launch_micro_batch(batch: int, micro_batch: int) -> typing.Awaitable[None]:
        micro_batch_id = batch * micro_batches + micro_batch

        return event_loop.create_task(
            client.run_mini_batch(micro_batch_id), name=f"micro batch {micro_batch_id} ({batch}-{micro_batch})"
        )

    for batch in range(5_000):
        optim.zero_grad()

        micro_batch_tasks = [launch_micro_batch(batch, micro_batch) for micro_batch in range(micro_batches)]
        await asyncio.gather(*micro_batch_tasks)

        optim.step()
        torch.cuda.empty_cache()


event_loop = asyncio.new_event_loop()

training_job = event_loop.create_task(run_training(), name="training")
event_loop.run_until_complete(training_job)
