# model parallelism

## B1: microbatch pipeline model parallelism

Implement pipeline parallelism with microbatches, as discussed during the lab.

As with the other data/model parallelism examples, you will need a Python script for the nodes and a shell script to orchestrate execution.

Be aware of the possibility of deadlocks: due to how `gloo` operates, it is possible to deadlock by having device 1 send $B_2$ to device 2 in the forward pass, and simultaneously, device 2 send $B_1$ in the backward pass.
Since both sends will await corresponding receives, the training will stop indefinitely.

Use `isend` & `irecv`, the asynchronous (non-blocking) versions of `send` & `recv` in `torch.distributed`.
Each call of the two function returns a `Work` object, on which calling `wait()` blocks, if needed, until the message exchange finishes.
Add comments or text explaining how you expect your implementation to work and test that it runs for the same number of steps and model architecture as in class.

Note that `torch.distributed`'s implementation of `gloo` does not currently support properly asynchronous communication even when using the corresponding primitives.
Thus, you will not see the same improvements in speed as with a backend like `nccl`.

You may also use the fact that `torch` gradients naturally accumulate if zeroed out.
Also, scaling the loss by a constant is equivalent to scaling the resulting gradients by the same constant.

You can rely on receiving messages in the same order they get sent between any device pair.
The `(i)send/recv` functions all support a `tag` attribute to match messages explicitly.
Although using it is good practice, it is not required.

You can refer to the [documentation](https://pytorch.org/docs/stable/distributed.html) and, if helpful, a related [tutorial](https://brsoff.github.io/tutorials/intermediate/dist_tuto.html) on the PyTorch website.

## B2: joint data & model parallelism

Implement a training setup that uses data and model parallelism together.

Create 2 pipelines of 3 stages running sequentially, where each stage works with 3 sequential micro-batches.

Once again, add comments or text explaining your implementation and test it on the setting that mimics those from the class.

You can use groups from `torch.distributed` to handle operations that require interaction between a subset of more than two but less than all workers.
