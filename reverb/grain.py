# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grain streams for live trajectories, timesteps, and structured patterns.

Import this module with the optional `grain` extra installed. Iterator state
cannot restore live sampling or pattern history. Use `prefetch` for bounded
background reads, and close the outer iterator when consumption stops.
"""

import collections
import copy
import dataclasses
import threading

import grain
import numpy as np
from reverb import pybind
from reverb import replay_dataset
from reverb import replay_sample
from reverb import structured_writer
import tensorflow as tf
import tree


class _Iterator(grain.DatasetIterator):

  def get_state(self):
    raise NotImplementedError("Live replay and pattern history cannot be checkpointed")

  def set_state(self, state):
    raise NotImplementedError("Construct an iterator to restart live sampling")

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.close()


class TrajectoryDataset(grain.IterDataset):
  """Batched trajectories with host `SampleInfo` and native sampling workers.

  Arguments match `reverb.ReplayDataset`. `max_samples` counts replay items,
  with `None` or `-1` denoting an unbounded stream. Without `batch_size`, each
  element is a trajectory with scalar metadata. `.batch` uses native batching.
  Optional `dtypes` and `shapes` describe and validate the output data structure.
  Rate-limiter timeouts end the sequence, including a final partial batch unless
  `drop_remainder` is set. Set `timeout_as_end_of_sequence=False` to raise instead.
  Each iterator owns a sampler, so workers must construct their own iterators.
  Grain `map`, `filter`, and `batch` compose with this source. Checkpointing and
  checkpoint-dependent prefetch transforms are unsupported. Use `prefetch`.
  """

  def __init__(self, server_address, table, batch_size=None, *,
               timeout_as_end_of_sequence=True, dtypes=None, shapes=None, **kwargs):
    super().__init__()
    self._spec = None
    if (dtypes is None) != (shapes is None):
      raise ValueError("Provide both `dtypes` and `shapes`")
    if dtypes is not None:
      self._spec = tree.map_structure_up_to(
          dtypes, lambda dtype, shape: tf.TensorSpec(shape, dtype),
          dtypes, shapes)
    self._unbatched = batch_size is None
    self._timeout_as_end = timeout_as_end_of_sequence
    if kwargs.get("max_samples") == -1:
      kwargs["max_samples"] = None
    self._config = replay_dataset.ReplayDataset(
        server_address, table, 1 if batch_size is None else batch_size, **kwargs)

  def batch(self, batch_size, *, drop_remainder=False, batch_fn=None):
    if not self._unbatched or batch_fn is not None:
      return super().batch(batch_size, drop_remainder=drop_remainder, batch_fn=batch_fn)
    result = copy.copy(self)
    result._unbatched = False
    result._config = dataclasses.replace(
        self._config, batch_size=batch_size, drop_remainder=drop_remainder)
    return result

  def __iter__(self):
    return _SampleIterator(self, timesteps=False)


class TimestepDataset(TrajectoryDataset):
  """Batched timesteps with repeated item metadata.

  Batches may span replay items. `max_samples` counts items, not timesteps.
  Item columns must have matching trajectory lengths and compatible step shapes.
  `drop_remainder` controls the last timestep batch after the item limit.
  """

  def __iter__(self):
    return _SampleIterator(self, timesteps=True)


class _SampleIterator(_Iterator):

  def __init__(self, dataset, timesteps):
    super().__init__()
    self._source = None
    self._unbatched = dataset._unbatched
    self._spec = dataset._spec
    self._source = replay_dataset._ReplayIterator(
        dataset._config, timesteps=timesteps, timeout_as_end=dataset._timeout_as_end)

  def __next__(self):
    if self._closed:
      raise StopIteration
    try:
      sample = next(self._source)
      if self._spec is not None:
        leaves = tree.flatten(sample.data)
        specs = tree.flatten(self._spec)
        if len(leaves) != len(specs):
          raise ValueError("Sample columns do not match `dtypes` and `shapes`")
        for value, spec in zip(leaves, specs):
          dtype = np.dtype(spec.dtype.as_numpy_dtype)
          if dtype.kind in "SU":
            dtype = np.dtype(object)
          shape_matches = (spec.shape.rank is None or
                           (spec.shape.rank == value.ndim - 1 and all(
                               expected is None or expected == actual
                               for expected, actual in zip(spec.shape, value.shape[1:]))))
          if value.dtype != dtype or not shape_matches:
            raise ValueError("Sample dtype or shape does not match the dataset specification")
        sample = replay_sample.ReplaySample(
            sample.info, tree.unflatten_as(self._spec, leaves))
      if self._unbatched:
        sample = tree.map_structure(lambda value: value[0], sample)
      return sample
    except BaseException:
      self.close()
      raise

  def close(self):
    self._closed = True
    if self._source is not None:
      self._source.close()


class PatternDataset(grain.IterDataset):
  """Apply Reverb patterns to a Grain stream of steps.

  Configurations come from `reverb.structured_writer.create_config`. Outputs
  contain the pattern's data structure. `is_end_of_episode` returns a scalar
  boolean for each step. An episode end applies end conditions before clearing
  history when `respect_episode_boundaries` is true. Input exhaustion does not
  imply an episode end. Configurations must produce compatible structures,
  shapes, and dtypes, as required by `structured_writer.infer_signature`.

  Set `input_batches=True` for a stream of blocks with a leading step dimension.
  The episode-end predicate then returns a boolean vector. Native processing
  consumes at most `input_block_size` steps per input conversion, and `.batch`
  assembles outputs in C++. Block boundaries do not imply episode boundaries.
  """

  def __init__(self, input_dataset, configs, respect_episode_boundaries,
               is_end_of_episode, *, input_batches=False, input_block_size=256):
    super().__init__(input_dataset)
    self._input_batches = input_batches
    self._block_size = replay_dataset._positive_integer(input_block_size, "input_block_size")
    self._output_batch_size = None
    self._drop_remainder = False
    configs = list(configs)
    if not configs:
      raise ValueError("`configs` must not be empty")
    self._configs = []
    self._history = 1
    for original in configs:
      config = structured_writer.Config()
      config.CopyFrom(original)
      history = max(self._history, max(
          (abs(min(node.start, node.stop)) for node in config.flat), default=1))
      config.conditions.add(buffer_length=True, ge=history)
      self._history = max(self._history, history)
      self._configs.append(config.SerializeToString())
    if any(c.pattern_structure != configs[0].pattern_structure for c in configs):
      raise ValueError("Patterns must have matching structures")
    self._structure = structured_writer.unpack_pattern(configs[0])
    self._respect = respect_episode_boundaries
    self._is_end = is_end_of_episode
    pybind.PatternWriter(self._configs, self._history)

  @classmethod
  def from_tensor_slices(cls, data, configs, respect_episode_boundaries,
                         is_end_of_episode, *, repeat=False, input_block_size=256):
    """Process array slices natively, with a vectorized episode-end predicate.

    Leaves share a leading step dimension. `repeat` repeats the input arrays,
    while episode boundaries remain controlled by `is_end_of_episode`.
    """
    source = grain.MapDataset.source([data])
    if repeat:
      source = source.repeat()
    source = source.to_iter_dataset(
        grain.ReadOptions(num_threads=0, prefetch_buffer_size=0))
    return cls(source, configs, respect_episode_boundaries, is_end_of_episode,
               input_batches=True, input_block_size=input_block_size)

  def batch(self, batch_size, *, drop_remainder=False, batch_fn=None):
    if not self._input_batches or self._output_batch_size is not None or batch_fn is not None:
      return super().batch(batch_size, drop_remainder=drop_remainder, batch_fn=batch_fn)
    result = copy.copy(self)
    result._output_batch_size = replay_dataset._positive_integer(batch_size, "batch_size")
    result._drop_remainder = drop_remainder
    return result

  def __iter__(self):
    if self._input_batches:
      return _BlockPatternIterator(iter(self._parents[0]), self)
    return _PatternIterator(iter(self._parents[0]), self)


class _BlockPatternIterator(_Iterator):

  def __init__(self, parent, dataset):
    super().__init__(parent)
    self._dataset = dataset
    self._writer = pybind.PatternWriter(dataset._configs, dataset._history)
    self._structure = None
    self._initialized = False
    self._block = None
    self._position = 0
    self._length = 0
    self._eof = False

  def _set_input(self):
    while self._position == self._length:
      block = next(self._parent)
      if self._closed:
        raise StopIteration
      if not self._initialized:
        self._structure = tree.map_structure(lambda _: None, block)
        self._initialized = True
        configs = [structured_writer.Config.FromString(c)
                   for c in self._dataset._configs]
        structured_writer.infer_signature(configs, tree.map_structure(
            lambda v: tf.TensorSpec(np.shape(v)[1:], np.asarray(v).dtype),
            block))
      tree.assert_same_structure(self._structure, block)
      columns = tree.flatten(block)
      if not columns or any(np.ndim(v) < 1 for v in columns):
        raise ValueError("Input blocks require a leading step dimension")
      self._length = len(columns[0])
      if any(len(v) != self._length for v in columns):
        raise ValueError("Input block columns must have matching lengths")
      self._block = block
      self._position = 0
    end = min(self._position + self._dataset._block_size, self._length)
    block = tree.map_structure(lambda v: v[self._position:end], self._block)
    flags = np.asarray(self._dataset._is_end(block))
    if flags.dtype != np.dtype(bool) or flags.shape != (end - self._position,):
      raise ValueError("`is_end_of_episode` must return a boolean vector for an input block")
    self._writer.SetBlock(tree.flatten(block), flags)
    self._position = end

  def __next__(self):
    size = self._dataset._output_batch_size or 1
    try:
      while not self._closed:
        values = self._writer.Read(size, self._dataset._respect, self._eof)
        if self._closed:
          raise StopIteration
        if values:
          if len(values[0]) < size and self._dataset._drop_remainder:
            raise StopIteration
          if self._dataset._output_batch_size is None:
            values = [value[0] for value in values]
          return tree.unflatten_as(self._dataset._structure, values)
        if self._eof:
          raise StopIteration
        try:
          self._set_input()
        except StopIteration:
          self._eof = True
      raise StopIteration
    except BaseException as error:
      cancelled = self._closed
      self.close()
      if cancelled and isinstance(error, RuntimeError):
        raise StopIteration from None
      raise

  def close(self):
    self._writer.Close()
    super().close()
    self._block = None


class _PatternIterator(_Iterator):

  def __init__(self, parent, dataset):
    super().__init__(parent)
    self._dataset = dataset
    self._writer = pybind.PatternWriter(dataset._configs, dataset._history)
    self._pending = collections.deque()
    self._structure = None
    self._initialized = False

  def __next__(self):
    if self._closed:
      raise StopIteration
    try:
      while not self._pending:
        step = next(self._parent)
        if not self._initialized:
          self._structure = tree.map_structure(lambda _: None, step)
          self._initialized = True
          configs = [structured_writer.Config.FromString(c)
                     for c in self._dataset._configs]
          structured_writer.infer_signature(configs, tree.map_structure(
              lambda v: tf.TensorSpec(np.shape(v), np.asarray(v).dtype),
              step))
        tree.assert_same_structure(self._structure, step)
        end = np.asarray(self._dataset._is_end(step))
        if end.shape != () or end.dtype != np.dtype(bool):
          raise ValueError("`is_end_of_episode` must return a scalar boolean")
        self._pending.extend(self._writer.Append(
            tree.flatten(step), bool(end), self._dataset._respect))
      return tree.unflatten_as(self._dataset._structure, self._pending.popleft())
    except BaseException:
      self.close()
      raise

  def close(self):
    super().close()
    self._pending.clear()
    self._writer = None


def pattern_dataset_with_info(*args, **kwargs):
  """Wrap pattern outputs in `ReplaySample` with zero metadata."""
  return PatternDataset(*args, **kwargs).map(
      lambda data: replay_sample.ReplaySample(replay_sample.SampleInfo.zeros(), data))


def prefetch(dataset, buffer_size=2):
  """Overlap bounded reads without checkpointing the live source.

  Closing the iterator cancels its source before joining the producer. Custom
  upstream iterators must support cancellation through `close`. Grain's
  checkpoint-dependent prefetch and multiprocessing transforms are unsupported.
  """
  return _PrefetchDataset(dataset, buffer_size)


class _PrefetchDataset(grain.IterDataset):

  def __init__(self, parent, buffer_size):
    super().__init__(parent)
    self._buffer_size = replay_dataset._positive_integer(buffer_size, "buffer_size")

  def __iter__(self):
    return _PrefetchIterator(iter(self._parents[0]), self._buffer_size)


class _PrefetchIterator(_Iterator):

  def __init__(self, parent, capacity):
    super().__init__(parent)
    self._capacity = capacity
    self._queue = collections.deque()
    self._condition = threading.Condition()
    self._done = False
    self._error = None
    self._thread = threading.Thread(target=self._produce, daemon=True)
    self._thread.start()

  def _produce(self):
    try:
      while True:
        with self._condition:
          self._condition.wait_for(
              lambda: self._closed or len(self._queue) < self._capacity)
          if self._closed:
            return
        value = next(self._parent)
        with self._condition:
          if self._closed:
            return
          self._queue.append(value)
          self._condition.notify_all()
    except StopIteration:
      pass
    except BaseException as error:
      with self._condition:
        self._error = error
    finally:
      with self._condition:
        self._done = True
        self._condition.notify_all()

  def __next__(self):
    with self._condition:
      self._condition.wait_for(lambda: self._closed or self._queue or self._done)
      if self._closed:
        raise StopIteration
      if self._queue:
        result = self._queue.popleft()
        self._condition.notify_all()
        return result
      error = self._error
    self.close()
    if error is not None:
      raise error
    raise StopIteration

  def close(self):
    with self._condition:
      self._closed = True
      self._queue.clear()
      self._condition.notify_all()
    self._parent.close()
    if threading.current_thread() is not self._thread:
      self._thread.join()
