# Goals
- Speed up training and inference 2-10x compared to baseline.
- Achieve high GPU utilization by offloading all scheduling of computation to
other threads/processes.
- Enable automatic tests of whether 

# Features
## Phase 1: initial
- Construct static computation graphs
- Merge computation graphs together to batch together operations which can be
computed together efficiently on the GPU.

## Phase 2: efficient training
Create merged computation graphs on worker processes and send them in a queue
to the main process to be executed on the GPU.

## Phase 3: batched inference
Batch together inference of multiple examples by processing multiple examples
concurrently (e.g. using coroutines), allowing each example to progress until
a decision is needed from the GPU, and only using the GPU in a batched manner
when all examples are awaiting results of a GPU computation.

## Phase 4: efficiency improvements for inference
Ensure high GPU utilization by having multiple batches undergoing inference
take turns using the GPU; while waiting for the GPU, they can perform CPU
computations to decide what next needs to be computed on the GPU.

## Phase 5: further optimizations
- Given multiple training examples, batch together ones which waste the least
amount of resources (e.g. sequences of similar lengths)
- Prune unnecessary parts of computation graph; sometimes we might define
computations and never use the results later
- Reuse computation graph if same computation is defined multiple times (e.g.
don't encode same schema multiple times)
- Avoid redundant `torch.stack` and `torch.unstack`

# Design
## `BatchedModule`
These are used as containers for `torch.nn.Module`s, calls to which will be
deferred (stored in the computation graph) and performed later in batches.
They can also contain other `BatchedModule`s as fields.

If instantiated when a `BatchingPolicy` is active, all `torch.nn.Module`
fields are replaced with stubs, which adds nodes to a computation graph when
called.

Requirements for component `torch.nn.Module`s:
- They should have `BatchKey`s defined that will specify which invocations
can be batched together.
- They should support computation in batches.
- They should avoid doing any work that's not directly involved in using the
GPU; that work should be placed outside in `BatchedModule`s.

## Computation graph
Calls to `torch.nn.Module`s inside a `BatchedModule` return `ResultHandle` objects.
`ResultHandle`s can be used as arguments to further `Module` calls.

To defer calls to other batchable operations, wrap them in `batching.batched_func`:
```python
batching.batched_func(torch.add)(x, y)
```

`ResultHandle`s can also be constructed for constant values through `batching.batched_value`.

## Scheduling
Calls to batchable operations consult `BatchingPolicy` to construct a `BatchKey` object.
Calls to the same operation with equal `BatchKey`s are batched together.

A `BatchKey` instance also specifies the following:
- How to invoke the underlying `Module` or function once all of the invocations to be batched together are available
- Properties and methods available on the corresponding `ResultHandle`:
  - How to iterate over the `ResultHandle`, to support `x, y = result_handle`.

Each batchable operation is assigned a `Node`. Once all examples to go in the
batch have been processed, all `Node`s are scheduled for computation with
agenda-based batching: the group of `Node`s with the same `BatchKey` and the
smallest mean depth are executed next.

## Execution
Using the serialized computation schedule, `Executor` looks up references to
the actual `torch.nn.Module`s, invokes them with the needed arguments, and
saves results for later.

# References
- [Static Automatic Batching in TensorFlow](http://proceedings.mlr.press/v97/agarwal19a/agarwal19a.pdf)
- [On-the-fly Operation Batching in Dynamic Computation Graphs](https://arxiv.org/abs/1705.07860)
- JAX
  - [Auto-vectorization with vmap](https://github.com/google/jax#auto-vectorization-with-vmap)