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

"""Bounded host input for compiled JAX learners."""

import collections
import concurrent.futures
import threading

import jax
import jax.numpy as jnp
import numpy as np


class Prefetch:
  """Own a cancellable source and prepare batches on a bounded worker queue."""

  def __init__(self, source, depth=2, transform=lambda x: x):
    if depth < 1:
      raise ValueError('`depth` must be positive')
    self.source = source
    self._transform = transform
    self._closed = threading.Event()
    self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    self._pending = collections.deque(
        self._pool.submit(self._read) for _ in range(depth))

  def _read(self):
    if self._closed.is_set():
      raise StopIteration
    return self._transform(next(self.source))

  def __iter__(self):
    return self

  def __next__(self):
    if self._closed.is_set():
      raise StopIteration
    try:
      value = self._pending.popleft().result()
      self._pending.append(self._pool.submit(self._read))
      return value
    except BaseException:
      self.close()
      raise

  def close(self):
    if not self._closed.is_set():
      self._closed.set()
      for future in self._pending:
        future.cancel()
      try:
        self.source.close()
      finally:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.close()


def update(params, data):
  """Fit a linear predictor to a constant target."""
  def loss(weights):
    return jnp.mean(jnp.square(data @ weights - 1.0))
  value, gradient = jax.value_and_grad(loss)(params)
  return params - 0.01 * gradient, value


def compile_step(data_spec, sharding=None):
  parameter_spec = jax.ShapeDtypeStruct((data_spec.shape[-1],), np.float32)
  kwargs = {}
  if sharding is not None:
    replicated = jax.sharding.NamedSharding(
        sharding.mesh, jax.sharding.PartitionSpec())
    kwargs = dict(in_shardings=(replicated, sharding),
                  out_shardings=(replicated, replicated))
  return jax.jit(update, donate_argnums=(0,), **kwargs).lower(
      parameter_spec, data_spec).compile()


def compile_scan(read_batch, data_spec, steps):
  """Compile host sampling with ordered effects and fixed batch shapes."""
  def train(params):
    def body(weights, _):
      data = jax.experimental.io_callback(
          read_batch, data_spec, ordered=True)
      return update(weights, data)
    return jax.lax.scan(body, params, None, length=steps)
  parameter_spec = jax.ShapeDtypeStruct((data_spec.shape[-1],), np.float32)
  return jax.jit(train, donate_argnums=(0,)).lower(parameter_spec).compile()


def local_sharding(batch_size, ndim):
  devices = jax.local_devices()
  if jax.process_count() != 1:
    raise ValueError('This example supports a single JAX process')
  if batch_size % len(devices):
    raise ValueError('`batch_size` must be divisible by the local device count')
  mesh = jax.sharding.Mesh(np.asarray(devices), ('data',))
  return jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('data', *([None] * (ndim - 1))))
