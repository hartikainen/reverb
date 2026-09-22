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

"""Bounded replay sampling for NumPy and JAX learners."""

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
  """A live replay stream with native prefetching and batching.

  Each iterator owns its sampler. Use it as a context manager to cancel blocked
  reads and release workers when training stops. Samples retain trajectory
  dimensions. A table signature restores nested data, otherwise data is a list.
  Corresponding data leaves must have identical shapes and dtypes within a batch.
  Numeric batches retain native storage as read-only NumPy views.
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
    max_samples_per_stream: Samples per RPC stream, or `-1` for automatic selection.
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
  max_samples_per_stream: int = -1

  def __post_init__(self):
    for name in ("batch_size", "prefetch_size", "signature_timeout_secs"):
      _positive_integer(getattr(self, name), name)
    for name in ("num_workers", "max_samples_per_stream"):
      if operator.index(getattr(self, name)) != -1:
        _positive_integer(getattr(self, name), name)
    if self.max_samples is not None and operator.index(self.max_samples) < 0:
      raise ValueError("`max_samples` must be nonnegative or `None`")
    if operator.index(self.rate_limiter_timeout_ms) < -1:
      raise ValueError("`rate_limiter_timeout_ms` must be at least `-1`")

  def __iter__(self):
    return _ReplayIterator(self)

  def as_jax_iterator(self, device=None):
    """Place batch data on a JAX device or sharding, retaining host metadata.

    JAX is an optional dependency. Data dtypes must be representable under its
    configured precision policy. Sample keys remain NumPy `uint64` arrays.
    """
    import jax  # pylint: disable=g-import-not-at-top
    return _DeviceIterator(iter(self), jax, device)


class _ReplayIterator:

  def __init__(self, dataset, *, timesteps=False, timeout_as_end=False):
    self._dataset = dataset
    self._timesteps = timesteps
    self._timeout_as_end = timeout_as_end
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
        dataset.max_samples_per_stream,
    )

  def __iter__(self):
    return self

  def __next__(self):
    if self._closed:
      raise StopIteration
    try:
      count = self._dataset.batch_size
      if self._timesteps:
        values = ([] if self._sampler is None else
                  self._sampler.GetNextTimestepBatch(count, self._timeout_as_end))
        count = len(values[0]) if values else 0
      else:
        if self._dataset.max_samples is not None:
          count = min(count, self._dataset.max_samples - self._count)
        values = (self._sampler.GetNextTrajectoryBatch(count, self._timeout_as_end)
                  if count else [])
        count = len(values[0]) if values else 0
      self._count += count
      if count == 0 or (count != self._dataset.batch_size and
                        self._dataset.drop_remainder):
        self.close()
        raise StopIteration
      if count < self._dataset.batch_size:
        self.close()
      info = replay_sample.SampleInfo(*values[:5])
      data = values[5:]
      if self._signature is not None:
        data = tree.unflatten_as(self._signature, data)
      return replay_sample.ReplaySample(info, data)
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


class _DeviceIterator:

  def __init__(self, source, jax, device):
    self._source = source
    self._jax = jax
    self._device = device

  def __iter__(self):
    return self

  def __next__(self):
    try:
      sample = next(self._source)
      for value in tree.flatten(sample.data):
        if self._jax.dtypes.canonicalize_dtype(value.dtype) != value.dtype:
          raise ValueError(
              f"JAX would change data dtype `{value.dtype}`; cast the data "
              "or enable `jax_enable_x64`")
      return replay_sample.ReplaySample(
          sample.info, self._jax.device_put(sample.data, self._device))
    except BaseException:
      self.close()
      raise

  def close(self):
    self._source.close()

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.close()
