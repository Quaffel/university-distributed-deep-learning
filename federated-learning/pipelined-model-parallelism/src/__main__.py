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

for itr in range(5_000):
    optim.zero_grad()
    # FORWARD PASS:
    client.forward()

    # BACKWARD PASS:
    client.backward()

    optim.step()
    torch.cuda.empty_cache()
