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

"""Exercise cancellation, compilation effects, and local-device sharding."""

import threading
import unittest

import input_pipeline
import jax
import numpy as np


class Source:
  def __init__(self, count=10):
    self.calls = 0
    self.count = count
    self.closed = False

  def __next__(self):
    if self.closed or self.calls == self.count:
      raise StopIteration
    self.calls += 1
    return np.full((4, 2, 3), self.calls / 10, np.float32)

  def close(self):
    self.closed = True


class InputPipelineTest(unittest.TestCase):
  def test_host_production_overlaps_device_transfer(self):
    produced_second = threading.Event()
    transfer_entered = threading.Event()
    finish_transfer = threading.Event()
    class Host(Source):
      def __next__(self):
        value = super().__next__()
        if self.calls == 2:
          produced_second.set()
        return value
    def transfer(value):
      transfer_entered.set()
      if not finish_transfer.wait(10):
        raise TimeoutError('Transfer was not released')
      return value
    source = Host(4)
    with input_pipeline.prefetch_to_device(
        source, depth=1, host_depth=1, transform=transfer) as batches:
      try:
        self.assertTrue(transfer_entered.wait(10))
        self.assertTrue(produced_second.wait(10))
      finally:
        finish_transfer.set()
      values = [float(batch[0, 0, 0]) for batch in batches]
    np.testing.assert_allclose(values, [.1, .2, .3, .4])
    self.assertTrue(source.closed)

  def test_staged_close_cancels_blocked_host(self):
    entered = threading.Event()
    released = threading.Event()
    class Blocked:
      def __next__(self):
        entered.set()
        if not released.wait(10):
          raise TimeoutError('Source cancellation failed')
        raise StopIteration
      def close(self):
        released.set()
    with input_pipeline.prefetch_to_device(Blocked()) as batches:
      self.assertTrue(entered.wait(10))
    self.assertTrue(released.is_set())
    with self.assertRaises(StopIteration):
      next(batches)

  def test_prefetch_preserves_order_and_exhaustion(self):
    source = Source(4)
    with input_pipeline.Prefetch(source, depth=2) as batches:
      values = [float(batch[0, 0, 0]) for batch in batches]
    np.testing.assert_allclose(values, [.1, .2, .3, .4])
    self.assertTrue(source.closed)

  def test_prefetch_propagates_failure(self):
    source = Source()
    def fail(_):
      raise ValueError('transform failed')
    with input_pipeline.Prefetch(source, transform=fail) as batches:
      with self.assertRaisesRegex(ValueError, 'transform failed'):
        next(batches)
    self.assertTrue(source.closed)

  def test_close_unblocks_worker(self):
    entered = threading.Event()
    released = threading.Event()
    class Blocked:
      def __next__(self):
        entered.set()
        if not released.wait(timeout=10):
          raise TimeoutError('Source cancellation failed')
        raise StopIteration
      def close(self):
        released.set()
    batches = input_pipeline.Prefetch(Blocked())
    self.assertTrue(entered.wait(timeout=10))
    batches.close()
    with self.assertRaises(StopIteration):
      next(batches)

  def test_scan_reads_at_runtime_on_each_invocation(self):
    source = Source()
    spec = jax.ShapeDtypeStruct((4, 2, 3), np.float32)
    train = input_pipeline.compile_scan(lambda: next(source), spec, 2)
    self.assertEqual(source.calls, 0)
    params = jax.device_put(np.zeros(3, np.float32))
    for expected_count in [2, 4]:
      params, losses = train(params)
      jax.block_until_ready((params, losses))
      self.assertEqual(source.calls, expected_count)
    expected = np.zeros(3, np.float32)
    reference = Source()
    for _ in range(4):
      expected, _ = input_pipeline.update(expected, next(reference))
    np.testing.assert_allclose(params, expected, rtol=1e-5)

  def test_sharded_update_matches_unsharded(self):
    spec = jax.ShapeDtypeStruct((4, 2, 3), np.float32)
    sharding = input_pipeline.local_sharding(4, 3)
    train = input_pipeline.compile_step(spec, sharding)
    replicated = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec())
    params = jax.device_put(np.zeros(3, np.float32), replicated)
    data = np.arange(24, dtype=np.float32).reshape(spec.shape) / 24
    result, _ = train(params, jax.device_put(data, sharding))
    expected, _ = input_pipeline.update(np.zeros(3, np.float32), data)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    self.assertEqual(len(sharding.device_set), jax.local_device_count())


if __name__ == '__main__':
  unittest.main()
