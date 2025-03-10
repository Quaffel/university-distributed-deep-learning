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
    async def run_mini_batch(self, micro_batch_id: int) -> None: ...

    @property
    def model(self) -> torch.nn.Module: ...


dmodel = 288
num_heads = 6
total_layers_count = 6
seq_l = 256


def build_dataset_iterator(
    tokenizer: AbstractTokenizer, batch_size: int
) -> typing.Iterator[torch.Tensor]:
    dataset = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)  # no skip
    return iter(dataset)


def as_asyncio_future(
    torch_task: torch.distributed.Work | None,
) -> asyncio.Future[None]:
    if torch_task is None:
        raise ValueError(
            "expected value other than None "
            "(was the operation destined at some process outside the current process group?)"
        )

    asyncio_future = asyncio.get_running_loop().create_future()

    backend = torch.distributed.get_backend().title().lower()
    if backend == torch.distributed.Backend.GLOO:
        # This is where the illusion of asynchrony falls apart: we don't actually yield execution
        # to some other task but await the result of the task synchronously and return an already completed future.
        #
        # torch.distributed's gloo backend does not support asynchronous IO operations.
        # It is reasonable to assume that gloo synchronizes all IO operations internally through a lock but still
        # expose all primitives that asynchronous code may need.
        # One such primitive are futures which allow asynchronous code to register callbacks that are called
        # as soon as the requested IO operation completed.
        # Unfortunately, calling 'torch.distributed.Work#get_future' when running with the gloo backend raises
        # an error indicating that the operation is not supported.
        # Because of this, we cannot represent the job as an asyncio future as we simply don't know when to complete it
        # (unless we regularly check for completion through polling or we wait synchronously).
        asyncio_future.set_result(None)
        torch_task.wait()
    else:
        torch_future = torch_task.get_future()
        torch_future.add_done_callback(lambda it: asyncio_future.set_result(it.value()))

    return asyncio_future


class Rank0Client(RankClient):
    def __init__(
        self,
        device: torch.device,
        subsequent_client: int,
        world_size: int,
        batch_size: int,
    ):
        self._device = device
        self._subsequent_client = subsequent_client
        self._batch_size = batch_size

        tokenizer = SPTokenizer()
        self._tokenizer = tokenizer
        self._model = LLamaFirstStage(
            tokenizer.vocab_size,
            dmodel=dmodel,
            num_heads=num_heads,
            device=device.type,
            n_layers=total_layers_count // world_size,
            ctx_size=seq_l,
        )

        self._loader = build_dataset_iterator(tokenizer, batch_size)

    async def run_mini_batch(self, micro_batch_id: int):
        out = next(self._loader)
        out = out.to(self._device)
        out = self._model.embed(out)

        await as_asyncio_future(
            dist.isend(out.to("cpu"), dst=self._subsequent_client, tag=micro_batch_id)
        )

        inp_grad = torch.empty((self._batch_size, seq_l, dmodel))
        await as_asyncio_future(
            dist.irecv(inp_grad, src=self._subsequent_client, tag=micro_batch_id)
        )

        out.backward(inp_grad.to(self._device))

    @property
    def model(self) -> torch.nn.Module:
        return self._model


class Rank1Client(RankClient):
    def __init__(
        self,
        device: torch.device,
        preceding_client: int,
        subsequent_client: int,
        world_size: int,
        batch_size: int,
    ) -> None:
        self._device = device
        self._preceding_client = preceding_client
        self._subsequent_client = subsequent_client
        self._batch_size = batch_size

        self._model = LLamaStage(
            dmodel=dmodel,
            num_heads=num_heads,
            device=device.type,
            n_layers=total_layers_count // world_size,
            ctx_size=seq_l,
        )

    async def run_mini_batch(self, micro_batch_id) -> None:
        inp_batch = torch.empty((self._batch_size, seq_l, dmodel))
        await as_asyncio_future(
            dist.irecv(inp_batch, src=self._preceding_client, tag=micro_batch_id)
        )

        with torch.no_grad():
            inp_batch = inp_batch.to(self._device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        out = self._model(inp_batch)

        await as_asyncio_future(
            dist.isend(out.to("cpu"), dst=self._subsequent_client, tag=micro_batch_id)
        )

        inp_grad = torch.empty((self._batch_size, seq_l, dmodel))
        await as_asyncio_future(
            dist.irecv(inp_grad, src=self._subsequent_client, tag=micro_batch_id)
        )

        out.backward(inp_grad.to(self._device))

        await as_asyncio_future(
            dist.isend(
                inp_batch.grad.to("cpu"), dst=self._preceding_client, tag=micro_batch_id
            )
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._model


class Rank2Client(RankClient):
    def __init__(
        self,
        device: torch.device,
        preceding_client: int,
        world_size: int,
        batch_size: int,
    ) -> None:
        self._device = device
        self._preceding_client = preceding_client
        self._batch_size = batch_size

        tokenizer = SPTokenizer()
        self._tokenizer = tokenizer
        self._model = LLamaLastStage(
            tokenizer.vocab_size,
            dmodel=dmodel,
            num_heads=num_heads,
            device=device.type,
            n_layers=total_layers_count // world_size,
            ctx_size=seq_l,
        )

        self._loader = build_dataset_iterator(tokenizer, batch_size)

    async def run_mini_batch(self, micro_batch_id: int) -> None:
        target = next(self._loader)
        inp_batch = torch.empty((self._batch_size, seq_l, dmodel))
        await as_asyncio_future(
            dist.irecv(inp_batch, src=self._preceding_client, tag=micro_batch_id)
        )

        with torch.no_grad():
            inp_batch = inp_batch.to(self._device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        logits = self._model(inp_batch)
        loss = causalLLMLoss(logits, target, self._tokenizer.vocab_size)
        loss.backward()

        print(f"microbatch {micro_batch_id}: loss {loss.item()}")

        await as_asyncio_future(
            dist.isend(
                inp_batch.grad.to("cpu"), dst=self._preceding_client, tag=micro_batch_id
            )
        )

    @property
    def model(self) -> torch.nn.Module:
        return self._model
