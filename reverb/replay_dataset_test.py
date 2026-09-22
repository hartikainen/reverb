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

"""Replay stream batching, cancellation, and learner integration tests."""

import threading
import unittest

import numpy as np
import reverb


class ReplayDatasetTest(unittest.TestCase):

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

  def test_batch_and_remainder(self):
    for value in range(5):
      self.client.insert(np.float32(value), {"data": 1.0})
    with iter(self.dataset(max_samples=5, drop_remainder=False)) as batches:
      values = [batch.data[0].reshape(-1).tolist() for batch in batches]
    self.assertEqual(values, [[0.0, 1.0], [2.0, 3.0], [4.0]])

  def test_drop_remainder(self):
    for value in range(3):
      self.client.insert(np.float32(value), {"data": 1.0})
    with iter(self.dataset(max_samples=3)) as batches:
      self.assertEqual(len(list(batches)), 1)

  def test_close_cancels_empty_table_read(self):
    iterator = iter(self.dataset())
    finished = threading.Event()
    def read():
      try:
        next(iterator)
      except (reverb.ReverbError, StopIteration):
        pass
      finally:
        finished.set()
    thread = threading.Thread(target=read)
    thread.start()
    iterator.close()
    thread.join(timeout=10)
    self.assertTrue(finished.is_set())

  def test_empty_stream(self):
    with iter(self.dataset(max_samples=0)) as iterator:
      self.assertEqual(list(iterator), [])

  def test_timeout_is_propagated(self):
    with iter(self.dataset(rate_limiter_timeout_ms=10)) as iterator:
      with self.assertRaises(reverb.DeadlineExceededError):
        next(iterator)


if __name__ == "__main__":
  unittest.main()
