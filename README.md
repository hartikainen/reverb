# Reverb
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dm-reverb)
[![PyPI version](https://badge.fury.io/py/dm-reverb.svg)](https://badge.fury.io/py/dm-reverb)

Reverb is an efficient and easy-to-use data storage and transport system
designed for machine learning research. Reverb is primarily used as an
experience replay system for distributed reinforcement learning algorithms but
the system also supports multiple data structure representations such as FIFO,
LIFO, and priority queues.

## Table of Contents

-   [Installation](#installation)
-   [Quick Start](#quick-start)
-   [Detailed Overview](#detailed-overview)
    -   [Tables](#tables)
    -   [Item Selection Strategies](#item-selection-strategies)
    -   [Rate Limiting](#rate-limiting)
    -   [Sharding](#sharding)
    -   [Checkpointing](#checkpointing)
-   [Citation](#citation)

## Installation

Please keep in mind that Reverb is not hardened for production use, and while we
do our best to keep things in working order, things may break or segfault.

> :warning: Reverb currently only supports Linux based OSes.

The recommended way to install Reverb is with `pip`. We also provide instructions
to build from source using the same docker images we use for releases.

TensorFlow can be installed separately or as part of the `pip` install.
Installing TensorFlow as part of the install ensures compatibility.

```shell
$ pip install dm-reverb[tensorflow]

# Without Tensorflow install and version dependency check.
$ pip install dm-reverb
```

### Nightly builds

[![PyPI version](https://badge.fury.io/py/dm-reverb-nightly.svg)](https://badge.fury.io/py/dm-reverb-nightly)

```shell
$ pip install dm-reverb-nightly[tensorflow]

# Without Tensorflow install and version dependency check.
$ pip install dm-reverb-nightly

```

### Build from source

[This guide](reverb/pip_package/README.md)
details how to build Reverb from source.

#### Bzlmod source targets

The Bzlmod source configuration selects Bazel `9.2.0`, Python `3.13`, and
TensorFlow `2.21.0`. It exposes the Python library `//reverb:reverb`
and the server executable `//reverb/server_executable:server_main`:

```console
bazel build \
  //reverb:reverb //reverb/server_executable:server_main
bazel test \
  //reverb:pybind_test //reverb:trajectory_writer_test
```

`MODULE.bazel` pins the native dependencies to TensorFlow's ABI. Python packages
and TensorFlow schema sources have checksums. The dependency declarations and
build helpers are contained in this repository.

A consuming Bazel module can depend on `@reverb//reverb:reverb`. Its root module
must select a supported Python toolchain and apply the native version constraints
and dependency patches declared by `single_version_override` and
`archive_override` in `MODULE.bazel`. Bazel ignores overrides declared by
dependency modules. Copy the patches from `third_party/bzlmod` into the consuming
root and use root-local patch labels in those overrides. The consuming build
also needs `--incompatible_autoload_externally=+@rules_cc` for the pinned native
dependencies, and the gRPC settings in `.bazelrc`.

The native Protobuf module `reverb_protobuf` matches TensorFlow's ABI. The
`protobuf` module supplies Bazel's protocol rules and providers. Their versions
are independent because upgrading build rules must not change TensorFlow's
native runtime.

Wheel packaging uses the same module graph as the source targets. See the
[wheel build guide](reverb/pip_package/README.md).


### Reverb Releases

Due to some underlying libraries such as `protoc` and `absl`, Reverb has to be
paired with a specific version of TensorFlow. If installing Reverb as
`pip install dm-reverb[tensorflow]` the correct version of Tensorflow will be
installed. The table below lists the version of TensorFlow that each release of
Reverb is associated with and some versions of interest:

  * 0.13.0 dropped Python 3.8 support.
  * 0.11.0 first version to support Python 3.11.
  * 0.10.0 last version to support Python 3.7.


Release | Branch / Tag                                               | TensorFlow Version
------- | ---------------------------------------------------------- | ------------------
Nightly | [master](https://github.com/deepmind/reverb)               | tf-nightly
0.14.0  | [v0.14.0](https://github.com/deepmind/reverb/tree/v0.14.0) | 2.14.0
0.13.0  | [v0.13.0](https://github.com/deepmind/reverb/tree/v0.13.0) | 2.14.0
0.12.0  | [v0.12.0](https://github.com/deepmind/reverb/tree/v0.12.0) | 2.13.0
0.11.0  | [v0.11.0](https://github.com/deepmind/reverb/tree/v0.11.0) | 2.12.0
0.10.0  | [v0.10.0](https://github.com/deepmind/reverb/tree/v0.10.0) | 2.11.0
0.9.0  | [v0.9.0](https://github.com/deepmind/reverb/tree/v0.9.0)   | 2.10.0
0.8.0  | [v0.8.0](https://github.com/deepmind/reverb/tree/v0.8.0)   | 2.9.0
0.7.x  | [v0.7.0](https://github.com/deepmind/reverb/tree/v0.7.0)   | 2.8.0

## Quick Start

Starting a Reverb server is as simple as:

```python
import reverb

server = reverb.Server(tables=[
    reverb.Table(
        name='my_table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1)),
    ],
)
```

Create a client to communicate with the server:

```python
client = reverb.Client(f'localhost:{server.port}')
print(client.server_info())
```

Write some data to the table:

```python
# Creates a single item and data element [0, 1].
client.insert([0, 1], priorities={'my_table': 1.0})
```

An item can also reference multiple data elements:

```python
# Appends three data elements and inserts a single item which references all
# of them as {'a': [2, 3, 4], 'b': [12, 13, 14]}.
with client.trajectory_writer(num_keep_alive_refs=3) as writer:
  writer.append({'a': 2, 'b': 12})
  writer.append({'a': 3, 'b': 13})
  writer.append({'a': 4, 'b': 14})

  # Create an item referencing all the data.
  writer.create_item(
      table='my_table',
      priority=1.0,
      trajectory={
          'a': writer.history['a'][:],
          'b': writer.history['b'][:],
      })

  # Block until the item has been inserted and confirmed by the server.
  writer.flush()
```

The items we have added to Reverb can be read by sampling them:

```python
# client.sample() returns a generator.
print(list(client.sample('my_table', num_samples=2)))
```

Continue with the
[Reverb Tutorial](https://github.com/deepmind/reverb/tree/master/examples/demo.ipynb)
for an interactive tutorial.

## Detailed overview

Experience replay has become an important tool for training off-policy
reinforcement learning policies. It is used by algorithms such as
[Deep Q-Networks (DQN)][DQN], [Soft Actor-Critic (SAC)][SAC],
[Deep Deterministic Policy Gradients (DDPG)][DDPG], and
[Hindsight Experience Replay][HER], ... However building an efficient, easy to
use, and scalable replay system can be challenging. For good performance Reverb
is implemented in C++ and to enable distributed usage it provides a gRPC service
for adding, sampling, and updating the contents of the tables. Python clients
expose the full functionality of the service in an easy to use fashion.
Furthermore native TensorFlow ops are available for performant integration with
TensorFlow and `tf.data`.

Although originally designed for off-policy reinforcement learning, Reverb's
flexibility makes it just as useful for on-policy reinforcement -- or even
(un)supervised learning. Creative users have even used Reverb to store and
distribute frequently updated data (such as model weights), acting as an
in-memory lightweight alternative to a distributed file system where each table
represents a file.

### Tables

A Reverb `Server` consists of one or more tables. A table holds items, and each
item references one or more data elements. Tables also define sample and
removal [selection strategies](#item-selection-strategies), a maximum item
capacity, and a [rate limiter](#rate-limiting).

Multiple items can reference the same data element, even if these items exist in
different tables. This is because items only contain references to data elements
(as opposed to a copy of the data itself). This also means that a data element
is only removed when there exists no item that contains a reference to it.

For example, it is possible to set up one Table as a Prioritized Experience
Replay (PER) for transitions (sequences of length 2), and another Table as a
(FIFO) queue of sequences of length 3. In this case the PER data could be used
to train DQN, and the FIFO data to train a transition model for the environment.

![Using multiple tables](docs/images/multiple_tables_example.png)

Items are automatically removed from the Table when one of two conditions are
met:

1.  Inserting a new item would cause the number of items in the Table to exceed
    its maximum capacity. Table's removal strategy is used to determine which
    item to remove.

1.  An item has been sampled more than the maximum number of times permitted by
    the Table's rate limiter. Such item is deleted.

Data elements not referenced anymore by any item are also deleted.

Users have full control over how data is sampled and removed from Reverb
tables. The behavior is primarily controlled by the
[item selection strategies](#item-selection-strategies) provided to the `Table`
as the `sampler` and `remover`. In combination with the
[`rate_limiter`](#rate-limiting) and `max_times_sampled`, a wide range of
behaviors can be achieved. Some commonly used configurations include:

**Uniform Experience Replay**

A set of `N=1000` most recently inserted items are maintained. By setting
`sampler=reverb.selectors.Uniform()`, the probability to select an item is the
same for all items. Due to `reverb.rate_limiters.MinSize(100)`, sampling
requests will block until 100 items have been inserted. By setting
`remover=reverb.selectors.Fifo()` when an item needs to be removed the oldest
item is removed first.

```python
reverb.Table(
     name='my_uniform_experience_replay_buffer',
     sampler=reverb.selectors.Uniform(),
     remover=reverb.selectors.Fifo(),
     max_size=1000,
     rate_limiter=reverb.rate_limiters.MinSize(100),
)
```

Examples of algorithms that make use of uniform experience replay include [SAC]
and [DDPG].

**Prioritized Experience Replay**

A set of `N=1000` most recently inserted items. By setting
`sampler=reverb.selectors.Prioritized(priority_exponent=0.8)`, the probability
to select an item is proportional to the item's priority.

Note: See [Schaul, Tom, et al.][PER] for the algorithm used in this
implementation of Prioritized Experience Replay.

```python
reverb.Table(
     name='my_prioritized_experience_replay_buffer',
     sampler=reverb.selectors.Prioritized(0.8),
     remover=reverb.selectors.Fifo(),
     max_size=1000,
     rate_limiter=reverb.rate_limiters.MinSize(100),
)
```

Examples of algorithms that make use of Prioritized Experience Replay are DQN
(and its variants), and
[Distributed Distributional Deterministic Policy Gradients][D4PG].

**Queue**

Collection of up to `N=1000` items where the oldest item is selected and removed
in the same operation. If the collection contains 1000 items then insert calls
are blocked until it is no longer full, if the collection is empty then sample
calls are blocked until there is at least one item.

```python
reverb.Table(
    name='my_queue',
    sampler=reverb.selectors.Fifo(),
    remover=reverb.selectors.Fifo(),
    max_size=1000,
    max_times_sampled=1,
    rate_limiter=reverb.rate_limiters.Queue(size=1000),
)

# Or use the helper classmethod `.queue`.
reverb.Table.queue(name='my_queue', max_size=1000)
```

Examples of algorithms that make use of Queues are
[IMPALA](https://arxiv.org/abs/1802.01561) and asynchronous implementations of
[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).

### Item selection strategies

Reverb defines several selectors that can be used for item sampling or removal:

-   **Uniform:** Sample uniformly among all items.
-   **Prioritized:** Samples proportional to stored priorities.
-   **FIFO:** Selects the oldest data.
-   **LIFO:** Selects the newest data.
-   **MinHeap:** Selects data with the lowest priority.
-   **MaxHeap:** Selects data with the highest priority.

Any of these strategies can be used for sampling or removing items from a
Table. This gives users the flexibility to create customized Tables that best
fit their needs.

### Rate Limiting

Rate limiters allow users to enforce conditions on when items can be inserted
and/or sampled from a Table. Here is a list of the rate limiters that are
currently available in Reverb:

-   **MinSize:** Sets a minimum number of items that must be in the Table before
    anything can be sampled.
-   **SampleToInsertRatio:** Sets that the average ratio of inserts to samples
    by blocking insert and/or sample requests. This is useful for controlling
    the number of times each item is sampled before being removed.
-   **Queue:** Items are sampled exactly once before being removed.
-   **Stack:** Items are sampled exactly once before being removed.

### Sharding

Reverb servers are unaware of each other and when scaling up a system to a multi
server setup data is not replicated across more than one node. This makes Reverb
unsuitable as a traditional database but has the benefit of making it trivial to
scale up systems where some level of data loss is acceptable.

Distributed systems can be horizontally scaled by simply increasing the number
of Reverb servers. When used in combination with a gRPC compatible load
balancer, the address of the load balanced target can simply be provided to a
Reverb client and operations will automatically be distributed across the
different nodes. You'll find details about the specific behaviors in the
documentation of the relevant methods and classes.

If a load balancer is not available in your setup or if more control is required
then systems can still be scaled in almost the same way. Simply increase the
number of Reverb servers and create separate clients for each server.

### Checkpointing

Reverb supports checkpointing; the state and content of Reverb servers can be
stored to permanent storage. While checkpointing, the `Server` serializes all of
its data and metadata needed to reconstruct it. During this process the `Server`
blocks all incoming insert, sample, update, and delete requests.

Checkpointing is done with a call from the Reverb `Client`:

```python
# client.checkpoint() returns the path the checkpoint was written to.
checkpoint_path = client.checkpoint()
```

To restore the `reverb.Server` from a checkpoint:

```python
# The checkpointer accepts the path of the root directory in which checkpoints
# are written. If we pass the root directory of the checkpoints written above
# then the new server will load the most recent checkpoint written from the old
# server.
checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
  path=checkpoint_path.rsplit('/', 1)[0])

# The arguments passed to `tables=` must be the same as those used by the
# `Server` that wrote the checkpoint.
server = reverb.Server(tables=[...], checkpointer=checkpointer)
```

Refer to
[tfrecord_checkpointer.h](https://github.com/deepmind/reverb/tree/master/reverb/cc/platform/tfrecord_checkpointer.h)
for details on the implementation of checkpointing in Reverb.

## Starting Reverb using `reverb_server` (beta)

Installing `dm-reverb` using `pip` will install a `reverb_server` script, which
accepts its config as a textproto. For example:

```bash
$ reverb_server --config="
port: 8000
tables: {
  table_name: \"my_table\"
  sampler: {
    fifo: true
  }
  remover: {
    fifo: true
  }
  max_size: 200 max_times_sampled: 5
  rate_limiter: {
    min_size_to_sample: 1
    samples_per_insert: 1
    min_diff: $(python3 -c "import sys; print(-sys.float_info.max)")
    max_diff: $(python3 -c "import sys; print(sys.float_info.max)")
  }
}"
```

The `rate_limiter` config is equivalent to the Python expression `MinSize(1)`,
see `rate_limiters.py`.


## Citation

If you use this code, please cite the
[Reverb paper](https://arxiv.org/abs/2102.04736) as

```
@misc{cassirer2021reverb,
      title={Reverb: A Framework For Experience Replay},
      author={Albin Cassirer and Gabriel Barth-Maron and Eugene Brevdo and Sabela Ramos and Toby Boyd and Thibault Sottiaux and Manuel Kroiss},
      year={2021},
      eprint={2102.04736},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!-- Links to papers go here -->

[D4PG]: https://arxiv.org/abs/1804.08617
[DDPG]: https://arxiv.org/abs/1509.02971
[DQN]: https://www.nature.com/articles/nature14236
[HER]: https://arxiv.org/abs/1707.01495
[PER]: https://arxiv.org/abs/1511.05952
[SAC]: https://arxiv.org/abs/1801.01290

### JAX learner input

`ReplayDataset` assembles trajectory data and metadata in Reverb's bounded native
sampler with the GIL released, then exposes a batch of NumPy arrays. Corresponding
data leaves must have identical shapes and dtypes within each batch. Use a context
manager so early termination cancels blocked reads and releases sampler workers:

```python
import reverb

source = reverb.ReplayDataset("localhost:8000", "experience", batch_size=256)
with source.as_jax_iterator() as batches:
    for sample in batches:
        state, priorities = learner_step(state, sample.data)
        client.mutate_priorities(
            "experience", dict(zip(sample.info.key, priorities)))
```

Install the `jax` wheel extra, or depend on `@reverb//reverb:jax` from a
Bazel Python target, to use `as_jax_iterator`. Only `sample.data` moves
to the device. Metadata remains on the host, preserving `uint64` replay keys.
Data that would lose dtype precision under JAX's configuration raises an error.
The iterator consumes live replay state and does not promise deterministic
checkpoint restoration. Each learner process should create its own iterator.
Use `tf.TensorSpec(shape, dtype)` for table signatures. The NumPy and
Grain input paths retain Reverb's native TensorFlow dependency.

### Grain datasets

Install `dm-reverb[grain,jax]`, or depend on `@reverb//reverb:grain` and
`@reverb//reverb:jax`. Grain `0.2.18` requires Python `3.11` or later.
Importing `reverb` does not import Grain or JAX.

```python
from reverb import grain as reverb_grain

source = reverb_grain.TrajectoryDataset(
    "localhost:8000", "experience", batch_size=256)
with iter(reverb_grain.prefetch(source, buffer_size=2)) as batches:
    for sample in batches:
        state = learner_step(state, sample.data)
```

`TrajectoryDataset` and `TimestepDataset` are Grain `IterDataset` sources.
Both return individual elements with scalar NumPy metadata by default.
`.batch` fuses sampling and batching in C++, or `batch_size` selects batching at
construction. Trajectories retain the time dimension. Timesteps flatten each item's
trajectory and repeat its metadata, with batches allowed to cross item boundaries.
`max_samples` counts replay items in both interfaces. `drop_remainder` applies to
output batches. Within a timestep item, columns must have matching lengths.
Rate-limiter timeouts end the sequence and allow a final partial batch when
`drop_remainder=False`. Set `timeout_as_end_of_sequence=False` to raise a timeout
error. `dtypes` and `shapes` optionally validate and restore an explicit data
structure, including tables without signatures. `max_samples_per_stream` controls
RPC stream rotation, and `num_workers=-1` requests automatic worker selection.

`PatternDataset(input_dataset, configs, respect_episode_boundaries,
is_end_of_episode)` applies `structured_writer.create_config` patterns to a Grain
stream of steps. It uses Reverb's native pattern conditions, slices, and episode
handling, and returns the pattern's nested data structure. `pattern_dataset_with_info`
adds zero-valued metadata. Input exhaustion does not implicitly end an episode.

Use `PatternDataset.from_tensor_slices` for resident arrays, or
`input_batches=True` for a Grain source of step blocks. The episode-end predicate
then operates on a block and returns a boolean vector. Pattern processing runs
in C++ with the GIL released, and `.batch` assembles the output arrays natively.
`input_block_size` bounds the steps converted into native storage at a time.
Input blocks may cross episode boundaries, and patterns may cross block boundaries.

```python
from reverb import grain as reverb_grain
from reverb import structured_writer

reference = structured_writer.create_reference_step(
    {"observation": None, "is_last": None})
config = structured_writer.create_config(
    {"observation": reference["observation"][-8:]}, "unused")
steps = reverb_grain.TimestepDataset(
    "localhost:8000", "experience").batch(256).map(lambda sample: sample.data)
windows = reverb_grain.PatternDataset(
    steps, [config], respect_episode_boundaries=True,
    is_end_of_episode=lambda block: block["is_last"],
    input_batches=True).batch(64, drop_remainder=True)
with iter(reverb_grain.prefetch(windows)) as batches:
    for batch in batches:
        state = learner_step(state, batch)
```

This example requires the table signature to expose `observation` and `is_last`.
The step-wise constructor also accepts arbitrary Python predicates and Grain
transforms. That path performs Python work per step. Select a block source when
comparing with a TensorFlow pipeline whose input and pattern loop stay native.

Grain `map`, `filter`, and `batch` compose with these datasets. Use
`reverb_grain.prefetch` for a bounded background queue that cancels its upstream
sampler before joining its worker. Close the outer iterator on early termination.
Each iterator owns its sampler or pattern history. Construct iterators inside
learner processes instead of sharing them across processes. Independent samplers
read live server state, so restart does not replay the same sequence. Iterator
checkpoint restoration, Grain's checkpoint-dependent prefetch transforms, and
Grain multiprocessing transforms are unsupported.

`//examples:dataset_semantics_test` checks Grain against outputs generated by
TensorFlow `2.21.0` and Reverb `928a50dbdc77`. The fixture covers pattern conditions,
multiple configurations, episode boundaries, data, metadata, sampling limits,
timeouts, and batch remainders. Run `examples/dataset_semantics.py` with
`--backend=tf` or `--backend=grain` and `--output` in the corresponding wheel
environment to repeat the differential check.
