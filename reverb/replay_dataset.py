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

"""Bounded replay sampling for NumPy learners."""

import dataclasses
import operator

import numpy as np
from reverb import client as client_lib
from reverb import replay_sample
import tree


def _positive_integer(value, name):
  value = operator.index(value)
  if value <= 0:
    raise ValueError(f"`{name}` must be positive")
  return value


@dataclasses.dataclass(frozen=True)
class ReplayDataset:
  """A live replay stream with native prefetching and NumPy batching.

  Each iterator owns its sampler. Use it as a context manager to cancel blocked
  reads and release workers when training stops. Samples retain trajectory
  dimensions. A table signature restores nested data, otherwise data is a list.
  Iterators do not support checkpoint restoration or deterministic replay.

  Attributes:
    server_address: Reverb server address.
    table: Table to sample.
    batch_size: Samples per batch.
    max_samples: Sample limit, or `None` for an unbounded stream.
    drop_remainder: Discard a final incomplete batch.
    prefetch_size: Maximum in-flight samples per native worker.
    num_workers: Native sampler worker count.
    rate_limiter_timeout_ms: Server sampling timeout, or `-1` to wait.
    signature_timeout_secs: Timeout for fetching the table signature.
  """

  server_address: str
  table: str
  batch_size: int
  max_samples: int | None = None
  drop_remainder: bool = True
  prefetch_size: int = 32
  num_workers: int = 1
  rate_limiter_timeout_ms: int = -1
  signature_timeout_secs: int = 30

  def __post_init__(self):
    for name in ("batch_size", "prefetch_size", "num_workers", "signature_timeout_secs"):
      _positive_integer(getattr(self, name), name)
    if self.max_samples is not None and operator.index(self.max_samples) < 0:
      raise ValueError("`max_samples` must be nonnegative or `None`")
    if operator.index(self.rate_limiter_timeout_ms) < -1:
      raise ValueError("`rate_limiter_timeout_ms` must be at least `-1`")

  def __iter__(self):
    return _ReplayIterator(self)



class _ReplayIterator:

  def __init__(self, dataset):
    self._dataset = dataset
    self._closed = False
    self._count = 0
    self._sampler = None
    self._signature = None
    if dataset.max_samples == 0:
      return
    client = client_lib.Client(dataset.server_address)
    self._signature = client.server_info(
        timeout=dataset.signature_timeout_secs)[dataset.table].signature
    self._sampler = client._client.NewSampler(  # pylint: disable=protected-access
        dataset.table,
        -1 if dataset.max_samples is None else dataset.max_samples,
        dataset.prefetch_size,
        dataset.num_workers,
        dataset.rate_limiter_timeout_ms,
    )

  def __iter__(self):
    return self

  def __next__(self):
    if self._closed:
      raise StopIteration
    samples = []
    try:
      for _ in range(self._dataset.batch_size):
        if (self._dataset.max_samples is not None
            and self._count >= self._dataset.max_samples):
          break
        values = self._sampler.GetNextTrajectory()
        info = replay_sample.SampleInfo(*values[:5])
        data = values[5:]
        if self._signature is not None:
          data = tree.unflatten_as(self._signature, data)
        samples.append(replay_sample.ReplaySample(info, data))
        self._count += 1
      if not samples or (
          len(samples) != self._dataset.batch_size
          and self._dataset.drop_remainder):
        self.close()
        raise StopIteration
      return tree.map_structure(lambda *xs: np.stack(xs), *samples)
    except BaseException as error:
      cancelled = self._closed
      self.close()
      if cancelled and isinstance(error, RuntimeError):
        raise StopIteration from None
      raise

  def close(self):
    if not self._closed:
      self._closed = True
      if self._sampler is not None:
        self._sampler.Close()

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.close()
