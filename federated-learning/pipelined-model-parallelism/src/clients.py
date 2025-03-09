import asyncio
import typing

import torch
import torch.distributed as dist
from simplellm.dataloaders import TinyStories  # get our dataset
from simplellm.llama import LLamaLastStage  # get our models
from simplellm.llama import LLamaFirstStage, LLamaStage
from simplellm.losses import causalLLMLoss  # our loss
from simplellm.tokenizers import SPTokenizer  # get our tokenizer
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer


class RankClient(typing.Protocol):
    async def run_mini_batch(self) -> None: ...

    @property
    def model(self) -> torch.nn.Module: ...


dmodel = 288
num_heads = 6
total_layers_count = 6
seq_l = 256

batch_size = 3


def build_dataset_iterator(
    tokenizer: AbstractTokenizer,
) -> typing.Iterator[torch.Tensor]:
    dataset = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)  # no skip
    return iter(dataset)


def as_asyncio_future[T](torch_future: torch.futures.Future[T]) -> asyncio.Future[T]:
    asyncio_future = asyncio.get_running_loop().create_future()
    torch_future.add_done_callback(lambda it: asyncio_future.set_result(it.value()))

    return asyncio_future


class Rank0Client(RankClient):
    def __init__(self, device: torch.device, world_size: int):
        tokenizer = SPTokenizer()

        self._device = device
        self._tokenizer = tokenizer
        self._model = LLamaFirstStage(
            tokenizer.vocab_size,
            dmodel=dmodel,
            num_heads=num_heads,
            device=device.type,
            n_layers=total_layers_count // world_size,
            ctx_size=seq_l,
        )

        self._loader = build_dataset_iterator(tokenizer)

    async def run_mini_batch(self):
        out = next(self._loader)
        out = out.to(self._device)
        out = self._model.embed(out)

        await as_asyncio_future(dist.isend(out.to("cpu"), 1).get_future())

        inp_grad = torch.empty((batch_size, seq_l, dmodel))
        await as_asyncio_future(dist.irecv(inp_grad, 1).get_future())

        out.backward(inp_grad.to(self._device))

    @property
    def model(self) -> torch.nn.Module:
        return self._model


class Rank1Client(RankClient):
    def __init__(self, device: torch.device, world_size: int) -> None:
        self._device = device
        self._model = LLamaStage(
            dmodel=dmodel,
            num_heads=num_heads,
            device=device.type,
            n_layers=total_layers_count // world_size,
            ctx_size=seq_l,
        )

    async def run_mini_batch(self) -> None:
        inp_batch = torch.empty((batch_size, seq_l, dmodel))
        await as_asyncio_future(dist.irecv(inp_batch, 0).get_future())

        with torch.no_grad():
            inp_batch = inp_batch.to(self._device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        out = self._model(inp_batch)

        await as_asyncio_future(dist.isend(out.to("cpu"), 2).get_future())

        inp_grad = torch.empty((batch_size, seq_l, dmodel))
        await as_asyncio_future(dist.irecv(inp_grad, 2).get_future())

        out.backward(inp_grad.to(self._device))

        await as_asyncio_future(dist.isend(inp_batch.grad.to("cpu"), 0))

    @property
    def model(self) -> torch.nn.Module:
        return self._model


class Rank2Client(RankClient):
    def __init__(self, device: torch.device, world_size: int) -> None:
        tokenizer = SPTokenizer()

        self._device = device
        self._tokenizer = tokenizer
        self._model = LLamaLastStage(
            tokenizer.vocab_size,
            dmodel=dmodel,
            num_heads=num_heads,
            device=device.type,
            n_layers=total_layers_count // world_size,
            ctx_size=seq_l,
        )

        self._loader = build_dataset_iterator(tokenizer)

    async def run_mini_batch(self) -> None:
        target = next(self._loader)
        inp_batch = torch.empty((batch_size, seq_l, dmodel))
        await as_asyncio_future(dist.irecv(inp_batch, 1).get_future())
        
        with torch.no_grad():
            inp_batch = inp_batch.to(self._device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        logits = self._model(inp_batch)
        loss = causalLLMLoss(logits, target, self._tokenizer.vocab_size)
        print(loss.item())
        loss.backward()

        await as_asyncio_future(dist.isend(inp_batch.grad.to("cpu"), 1).get_future())

    @property
    def model(self) -> torch.nn.Module:
        return self._model
