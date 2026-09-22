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

"""JAX device placement and host metadata tests."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import reverb
from reverb import replay_dataset


class JaxIteratorTest(unittest.TestCase):

  def setUp(self):
    self.server = reverb.Server([reverb.Table(
        "data", reverb.selectors.Fifo(), reverb.selectors.Fifo(), 100,
        reverb.rate_limiters.MinSize(1), max_times_sampled=1)])
    self.address = f"localhost:{self.server.port}"
    self.client = reverb.Client(self.address)

  def tearDown(self):
    self.server.stop()

  def dataset(self, **kwargs):
    return reverb.ReplayDataset(self.address, "data", batch_size=2, **kwargs)

  def test_precision_loss_closes_sampler(self):
    sample = reverb.ReplaySample(
        reverb.SampleInfo(0, 1., 1, 1., 1), np.array([1.], np.float64))
    source = mock.MagicMock()
    source.__next__.return_value = sample
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
      with self.assertRaisesRegex(ValueError, "would change data dtype"):
        next(replay_dataset._DeviceIterator(source, jax, None))
    finally:
      jax.config.update("jax_enable_x64", previous)
    source.close.assert_called_once()

  def test_jitted_learner_and_host_keys(self):
    for value in [2.0, 4.0]:
      self.client.insert(np.float32(value), {"data": 1.0})
    step = jax.jit(lambda weight, data: weight - 0.1 * jnp.mean(data))
    with self.dataset(max_samples=2).as_jax_iterator() as batches:
      batch = next(batches)
      self.assertIsInstance(batch.data[0], jax.Array)
      self.assertEqual(batch.info.key.dtype, np.dtype("uint64"))
      self.assertAlmostEqual(float(step(1.0, batch.data[0])), 0.7, places=5)

  def test_high_keys_are_not_placed_on_device(self):
    sample = reverb.ReplaySample(
        reverb.SampleInfo(np.array([2**64 - 1], np.uint64), 1., 1, 1., 1),
        np.array([1.], np.float32))
    source = mock.MagicMock()
    source.__next__.return_value = sample
    batch = next(replay_dataset._DeviceIterator(source, jax, None))
    self.assertEqual(int(batch.info.key[0]), 2**64 - 1)


if __name__ == "__main__":
  unittest.main()
