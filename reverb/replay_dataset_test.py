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

  def test_batch_preserves_dtypes_values_and_metadata(self):
    for dtype in [np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                  np.int32, np.uint32, np.int64, np.uint64, np.float16,
                  np.float32, np.float64, np.complex64, np.complex128,
                  np.dtype("bfloat16").type]:
      with self.subTest(dtype=dtype):
        inputs = [np.array([[0, 1], [1, 0]], dtype=dtype),
                  np.array([[1, 1], [0, 0]], dtype=dtype)]
        for value in inputs:
          self.client.insert(value, {"data": 3.0})
        with iter(self.dataset(max_samples=2)) as batches:
          batch = next(batches)
        np.testing.assert_array_equal(batch.data[0], np.stack(inputs)[:, None])
        self.assertEqual(batch.data[0].dtype, np.dtype(dtype))
        self.assertEqual(batch.info.key.dtype, np.dtype("uint64"))
        self.assertEqual(batch.info.probability.dtype, np.dtype("float64"))
        self.assertEqual(batch.info.table_size.dtype, np.dtype("int64"))
        self.assertEqual(batch.info.priority.dtype, np.dtype("float64"))
        self.assertEqual(batch.info.times_sampled.dtype, np.dtype("int32"))
        np.testing.assert_array_equal(batch.info.priority, [3.0, 3.0])
        np.testing.assert_array_equal(batch.info.times_sampled, [1, 1])
        self.assertEqual(batch.info.key.shape, (2,))
        self.assertNotEqual(batch.info.key[0], batch.info.key[1])

  def test_strings_and_empty_columns(self):
    for value in [b"a\x00b", b"c"]:
      self.client.insert([np.array(value, dtype=object),
                          np.empty((0, 3), np.float32)], {"data": 1.0})
    with iter(self.dataset(max_samples=2)) as batches:
      batch = next(batches)
    np.testing.assert_array_equal(batch.data[0], [[b"a\x00b"], [b"c"]])
    self.assertEqual(batch.data[1].shape, (2, 1, 0, 3))

  def test_mismatched_shapes_and_dtypes_close_iterator(self):
    for second in [np.zeros(3, np.float32), np.zeros(2, np.float64)]:
      with self.subTest(second=second):
        self.client.insert(np.zeros(2, np.float32), {"data": 1.0})
        self.client.insert(second, {"data": 1.0})
        with iter(self.dataset(max_samples=2)) as batches:
          with self.assertRaisesRegex(ValueError, "shapes and dtypes"):
            next(batches)
          with self.assertRaises(StopIteration):
            next(batches)

  def test_timeout_discards_partial_batch_and_closes_iterator(self):
    self.client.insert(np.float32(1), {"data": 1.0})
    with iter(self.dataset(rate_limiter_timeout_ms=100)) as batches:
      with self.assertRaises(reverb.DeadlineExceededError):
        next(batches)
      with self.assertRaises(StopIteration):
        next(batches)

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
