import asyncio
import os
from sys import argv

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


def _build_client(
    rank: int,
) -> clients.RankClient:
    if rank == 0:
        return clients.Rank0Client(device, world_size)
    elif rank == 1:
        return clients.Rank1Client(device, world_size)
    elif rank == 2:
        return clients.Rank2Client(device, world_size)
    else:
        raise ValueError(f"expected rank to be of value 0-2, got {rank}")


client = _build_client(rank)
optim = Adam(client.model.parameters(), lr=8e-4)

micro_batch_size = 1
micro_batches = 3

async def run_training():
    event_loop = asyncio.get_running_loop()

    for batch_id in range(5_000):
        optim.zero_grad()

        job = event_loop.create_task(client.run_mini_batch(), name=f"batch {batch_id}")

        await asyncio.gather(job)

        optim.step()
        torch.cuda.empty_cache()

event_loop = asyncio.new_event_loop()

training_job = event_loop.create_task(run_training(), name="training")
event_loop.run_until_complete(training_job)
