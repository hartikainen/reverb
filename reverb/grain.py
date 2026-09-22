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

  Arguments match `reverb.ReplayDataset`. `max_samples` counts replay items.
  Each iterator owns a sampler, so workers must construct their own iterators.
  Grain `map`, `filter`, and `batch` compose with this source. Checkpointing and
  checkpoint-dependent prefetch transforms are unsupported. Use `prefetch`.
  """

  def __init__(self, server_address, table, batch_size=1, **kwargs):
    super().__init__()
    self._config = replay_dataset.ReplayDataset(
        server_address, table, batch_size, **kwargs)

  def __iter__(self):
    return _SampleIterator(self._config, timesteps=False)


class TimestepDataset(TrajectoryDataset):
  """Batched timesteps with repeated item metadata.

  Batches may span replay items. `max_samples` counts items, not timesteps.
  Item columns must have matching trajectory lengths and compatible step shapes.
  `drop_remainder` controls the last timestep batch after the item limit.
  """

  def __iter__(self):
    return _SampleIterator(self._config, timesteps=True)


class _SampleIterator(_Iterator):

  def __init__(self, config, timesteps):
    super().__init__()
    self._source = None
    self._source = replay_dataset._ReplayIterator(config, timesteps=timesteps)

  def __next__(self):
    if self._closed:
      raise StopIteration
    try:
      return next(self._source)
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
  """

  def __init__(self, input_dataset, configs, respect_episode_boundaries,
               is_end_of_episode):
    super().__init__(input_dataset)
    configs = list(configs)
    if not configs:
      raise ValueError("`configs` must not be empty")
    self._configs = []
    self._history = 1
    for original in configs:
      config = structured_writer.Config()
      config.CopyFrom(original)
      history = max((abs(min(node.start, node.stop)) for node in config.flat),
                    default=1)
      config.conditions.add(buffer_length=True, ge=history)
      self._history = max(self._history, history)
      self._configs.append(config.SerializeToString())
    if any(c.pattern_structure != configs[0].pattern_structure for c in configs):
      raise ValueError("Patterns must have matching structures")
    self._structure = structured_writer.unpack_pattern(configs[0])
    self._respect = respect_episode_boundaries
    self._is_end = is_end_of_episode
    pybind.PatternWriter(self._configs, self._history)

  def __iter__(self):
    return _PatternIterator(iter(self._parents[0]), self)


class _PatternIterator(_Iterator):

  def __init__(self, parent, dataset):
    super().__init__(parent)
    self._dataset = dataset
    self._writer = pybind.PatternWriter(dataset._configs, dataset._history)
    self._pending = collections.deque()
    self._structure = None

  def __next__(self):
    if self._closed:
      raise StopIteration
    try:
      while not self._pending:
        step = next(self._parent)
        if self._structure is None:
          self._structure = tree.map_structure(lambda _: None, step)
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
