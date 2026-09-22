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

"""Grain replay composition, pattern semantics, and cancellation tests."""

import os
import pickle
import subprocess
import sys
import threading
import unittest

import grain
import numpy as np
import reverb
from reverb import grain as replay
from reverb import structured_writer as sw


def never_end(_):
  return False


class ImportTest(unittest.TestCase):

  def test_native_dependencies_are_isolated_in_both_import_orders(self):
    for imports in ['import grain; import reverb', 'import reverb; import grain']:
      with self.subTest(imports=imports):
        result = subprocess.run(
            [sys.executable, '-c', imports], capture_output=True, text=True,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class GrainTest(unittest.TestCase):

  def setUp(self):
    self.server = reverb.Server([reverb.Table(
        "data", reverb.selectors.Fifo(), reverb.selectors.Fifo(), 100,
        reverb.rate_limiters.MinSize(1), max_times_sampled=1)])
    self.addCleanup(self.server.stop)
    self.address = f"localhost:{self.server.port}"
    self.client = reverb.Client(self.address)

  def insert_trajectory(self, values):
    with self.client.trajectory_writer(num_keep_alive_refs=len(values)) as writer:
      for value in values:
        writer.append(np.int32(value))
      writer.create_item("data", 3.0, writer.history[:])
      writer.flush()

  def test_trajectory_composition_and_pickle(self):
    for value in range(6):
      self.client.insert(np.float32(value), {"data": 2.0})
    source = pickle.loads(pickle.dumps(replay.TrajectoryDataset(
        self.address, "data", 2, max_samples=6)))
    dataset = source.map(lambda sample: sample.data[0]).filter(
        lambda data: data[0, 0] > 0).batch(2)
    with iter(replay.prefetch(dataset, 1)) as iterator:
      np.testing.assert_array_equal(next(iterator), [[[2], [3]], [[4], [5]]])
      with self.assertRaises(StopIteration):
        next(iterator)

  def test_timestep_batch_crosses_items_and_preserves_metadata(self):
    self.insert_trajectory([1, 2])
    self.insert_trajectory([3, 4, 5])
    with iter(replay.TimestepDataset(
        self.address, "data", 3, max_samples=2, drop_remainder=False)) as iterator:
      batches = list(iterator)
    self.assertEqual([b.data[0].tolist() for b in batches], [[1, 2, 3], [4, 5]])
    self.assertEqual(batches[0].info.key.dtype, np.dtype("uint64"))
    keys = np.concatenate([b.info.key for b in batches])
    self.assertEqual(keys[0], keys[1])
    self.assertNotEqual(keys[1], keys[2])
    np.testing.assert_array_equal(keys[2:], [keys[2]] * 3)
    np.testing.assert_array_equal(batches[0].info.priority, [3, 3, 3])

  def test_timestep_drops_partial_batch(self):
    self.insert_trajectory([1, 2, 3])
    with iter(replay.TimestepDataset(
        self.address, "data", 2, max_samples=1)) as iterator:
      self.assertEqual([b.data[0].tolist() for b in iterator], [[1, 2]])

  def test_zero_items_and_checkpoint_rejection(self):
    for cls in [replay.TrajectoryDataset, replay.TimestepDataset]:
      with iter(cls(self.address, "data", max_samples=0)) as iterator:
        with self.assertRaises(NotImplementedError):
          iterator.get_state()
        with self.assertRaises(NotImplementedError):
          iterator.set_state({})
        self.assertEqual(list(iterator), [])

  def test_deadline_propagates_through_prefetch(self):
    source = replay.TimestepDataset(
        self.address, "data", rate_limiter_timeout_ms=20,
        timeout_as_end_of_sequence=False)
    with iter(replay.prefetch(source)) as iterator:
      with self.assertRaises(reverb.errors.DeadlineExceededError):
        next(iterator)
      self.assertFalse(iterator._thread.is_alive())

  def test_prefetch_close_cancels_blocked_producer_and_consumer(self):
    for cls in [replay.TrajectoryDataset, replay.TimestepDataset]:
      iterator = iter(replay.prefetch(cls(self.address, "data")))
      started = threading.Event()
      done = threading.Event()
      def consume():
        started.set()
        try:
          next(iterator)
        except StopIteration:
          done.set()
      consumer = threading.Thread(target=consume, daemon=True)
      consumer.start()
      self.assertTrue(started.wait(5))
      closer = threading.Thread(target=iterator.close, daemon=True)
      closer.start()
      closer.join(5)
      consumer.join(5)
      self.assertFalse(closer.is_alive())
      self.assertFalse(iterator._thread.is_alive())
      self.assertTrue(done.is_set())
      iterator.close()

  def test_unbatched_and_fused_batch_shapes(self):
    for cls, shape in [(replay.TrajectoryDataset, (3,)),
                       (replay.TimestepDataset, ())]:
      self.insert_trajectory([1, 2, 3])
      with iter(cls(self.address, "data", max_samples=1)) as iterator:
        sample = next(iterator)
        self.assertEqual(sample.data[0].shape, shape)
        self.assertEqual(sample.info.key.shape, ())
    self.insert_trajectory([1, 2, 3])
    with iter(replay.TimestepDataset(self.address, "data", max_samples=1).batch(2)) as iterator:
      self.assertEqual([x.data[0].tolist() for x in iterator], [[1, 2], [3]])

  def test_timeout_finishes_with_partial_batch(self):
    for cls in [replay.TrajectoryDataset, replay.TimestepDataset]:
      for drop in [False, True]:
        self.client.insert(np.int32(7), {"data": 1.0})
        source = cls(self.address, "data", rate_limiter_timeout_ms=50,
                     max_samples_per_stream=1).batch(2, drop_remainder=drop)
        with iter(source) as iterator:
          values = list(iterator)
        self.assertEqual(len(values), 0 if drop else 1)
        if not drop:
          self.assertEqual(values[0].data[0].reshape(-1).tolist(), [7])

  def test_explicit_spec_restores_structure_and_rejects_mismatch(self):
    self.insert_trajectory([1, 2, 3])
    source = replay.TimestepDataset(
        self.address, "data", max_samples=1, dtypes={"x": np.int32}, shapes={"x": ()})
    with iter(source.batch(2)) as iterator:
      self.assertEqual(next(iterator).data["x"].tolist(), [1, 2])
    self.insert_trajectory([1, 2, 3])
    with iter(replay.TrajectoryDataset(
        self.address, "data", max_samples=1, dtypes=np.float32, shapes=(3,))) as iterator:
      with self.assertRaisesRegex(ValueError, "dataset specification"):
        next(iterator)

  def test_independent_iterators(self):
    source = replay.TrajectoryDataset(self.address, "data", max_samples=1)
    first, second = iter(source), iter(source)
    first.close()
    self.client.insert(np.int32(7), {"data": 1.0})
    with second:
      self.assertEqual(next(second).data[0].item(), 7)


class PatternTest(unittest.TestCase):

  def dataset(self, values, pattern=None, conditions=(), respect=True, end=never_end):
    ref = sw.create_reference_step(0)
    config = sw.create_config(ref[-2:] if pattern is None else pattern(ref),
                              "unused", conditions)
    source = grain.MapDataset.source(values).to_iter_dataset()
    return replay.PatternDataset(source, [config], respect, end)

  def test_overlapping_history_and_squeezed_columns(self):
    dataset = self.dataset(list(range(5)),
                          lambda ref: {"window": ref[-3::2], "last": ref[-1]})
    with iter(dataset) as iterator:
      values = list(iterator)
    self.assertEqual([v["window"].tolist() for v in values], [[0, 2], [1, 3], [2, 4]])
    self.assertEqual([v["last"].item() for v in values], [2, 3, 4])

  def test_episode_boundaries(self):
    for respect, expected in [(True, [[0, 1], [2, 3]]),
                              (False, [[0, 1], [1, 2], [2, 3]])]:
      with iter(self.dataset(list(range(4)), respect=respect,
                             end=lambda x: x % 2 == 1)) as iterator:
        self.assertEqual([v.tolist() for v in iterator], expected)

  def test_end_condition_and_eof(self):
    with iter(self.dataset(list(range(5)), conditions=[sw.Condition.is_end_episode()],
                           end=lambda x: x == 3)) as iterator:
      self.assertEqual([v.tolist() for v in iterator], [[2, 3]])
    with iter(self.dataset([0, 1], conditions=[sw.Condition.is_end_episode()])) as iterator:
      self.assertEqual(list(iterator), [])

  def test_stride_condition(self):
    with iter(self.dataset(list(range(6)),
                           conditions=[sw.Condition.step_index() % 2 == 1])) as iterator:
      self.assertEqual([v.tolist() for v in iterator], [[0, 1], [2, 3], [4, 5]])

  def test_multiple_configs_do_not_mutate_input(self):
    ref = sw.create_reference_step(0)
    configs = [sw.create_config(ref[-2:], "unused"),
               sw.create_config(ref[-2:], "unused")]
    before = [c.SerializeToString() for c in configs]
    dataset = replay.PatternDataset(
        grain.MapDataset.source([0, 1, 2]).to_iter_dataset(), configs, True, never_end)
    self.assertEqual([c.SerializeToString() for c in configs], before)
    with iter(replay.prefetch(pickle.loads(pickle.dumps(dataset)))) as iterator:
      self.assertEqual([v.tolist() for v in iterator], [[0, 1], [0, 1], [1, 2], [1, 2]])

  def test_invalid_end_flag(self):
    for end in [lambda _: 1, lambda _: [True]]:
      with iter(self.dataset([0, 1], end=end)) as iterator:
        with self.assertRaisesRegex(ValueError, "scalar boolean"):
          next(iterator)

  def test_shape_change_fails(self):
    with iter(self.dataset([np.zeros(2), np.zeros(3)])) as iterator:
      with self.assertRaises((ValueError, RuntimeError)):
        next(iterator)

  def test_scalar_structure_cannot_change_to_container(self):
    with iter(self.dataset([np.int32(0), {"x": np.int32(1)}])) as iterator:
      with self.assertRaises((ValueError, TypeError)):
        next(iterator)

  def test_pattern_info_and_grain_batch(self):
    ref = sw.create_reference_step(0)
    dataset = replay.pattern_dataset_with_info(
        grain.MapDataset.source([0, 1, 2]).to_iter_dataset(),
        [sw.create_config(ref[-2:], "unused")], True, never_end).batch(2)
    iterator = iter(dataset)
    try:
      value = next(iterator)
      np.testing.assert_array_equal(value.data, [[0, 1], [1, 2]])
      np.testing.assert_array_equal(value.info.key, [0, 0])
    finally:
      iterator.close()


class BlockPatternTest(unittest.TestCase):

  def data(self):
    return {"value": np.arange(11, dtype=np.int32),
            "is_last": np.isin(np.arange(11), [3, 7, 10])}

  def configs(self, conditions=()):
    ref = sw.create_reference_step(self.data())
    return [sw.create_config({"window": ref["value"][-2:],
                              "last": ref["value"][-1]}, "unused", conditions)]

  def test_block_boundaries_and_native_batching_match_steps(self):
    data = self.data()
    for respect in [False, True]:
      for conditions in [[], [sw.Condition.is_end_episode()],
                         [sw.Condition.step_index() % 2 == 1]]:
        configs = self.configs(conditions)
        steps = [{key: value[i] for key, value in data.items()} for i in range(11)]
        reference = replay.PatternDataset(
            grain.MapDataset.source(steps).to_iter_dataset(), configs,
            respect, lambda step: step["is_last"])
        with iter(reference) as iterator:
          expected = list(iterator)
        for block_size in [1, 3, 8]:
          for batch_size in [1, 2, 7]:
            with self.subTest(respect=respect, block=block_size, batch=batch_size):
              dataset = replay.PatternDataset.from_tensor_slices(
                  data, configs, respect, lambda block: block["is_last"],
                  input_block_size=block_size).batch(batch_size)
              with iter(dataset) as iterator:
                actual = list(iterator)
              for key in ["window", "last"]:
                np.testing.assert_array_equal(
                    np.concatenate([x[key] for x in actual]),
                    np.stack([x[key] for x in expected]))

  def test_eof_does_not_end_episode_and_drop_remainder(self):
    data = self.data()
    data["is_last"][:] = False
    dataset = replay.PatternDataset.from_tensor_slices(
        data, self.configs([sw.Condition.is_end_episode()]), True,
        lambda block: block["is_last"])
    with iter(dataset.batch(3)) as iterator:
      self.assertEqual(list(iterator), [])
    dataset = replay.PatternDataset.from_tensor_slices(
        data, self.configs(), True, lambda block: block["is_last"])
    with iter(dataset.batch(3, drop_remainder=True)) as iterator:
      batches = list(iterator)
    self.assertEqual(len(batches), 3)
    self.assertEqual(batches[-1]["last"].tolist(), [7, 8, 9])

  def test_multiple_configs_and_unbatched_outputs(self):
    configs = self.configs() * 2
    dataset = replay.PatternDataset.from_tensor_slices(
        self.data(), configs, True, lambda block: block["is_last"], input_block_size=3)
    with iter(dataset) as iterator:
      first, second = next(iterator), next(iterator)
    self.assertEqual(first["last"].shape, ())
    np.testing.assert_array_equal(first["window"], second["window"])
    iterator = iter(dataset.batch(5).batch(2))
    try:
      self.assertEqual(next(iterator)["window"].shape, (2, 5, 2))
    finally:
      iterator.close()

  def test_stream_of_blocks_and_empty_blocks(self):
    data = self.data()
    blocks = [{k: v[:0] for k, v in data.items()},
              {k: v[:5] for k, v in data.items()},
              {k: v[5:] for k, v in data.items()}]
    dataset = replay.PatternDataset(
        grain.MapDataset.source(blocks).to_iter_dataset(), self.configs(),
        True, lambda block: block["is_last"], input_batches=True).batch(2)
    with iter(dataset) as iterator:
      values = np.concatenate([x["last"] for x in iterator])
    np.testing.assert_array_equal(values, [1, 2, 3, 5, 6, 7, 9, 10])

  def test_vectorized_predicate_validation(self):
    for predicate in [lambda _: True, lambda _: np.zeros(11, np.int32),
                      lambda _: np.zeros(3, bool)]:
      dataset = replay.PatternDataset.from_tensor_slices(
          self.data(), self.configs(), True, predicate)
      with iter(dataset) as iterator:
        with self.assertRaisesRegex(ValueError, "boolean vector"):
          next(iterator)

  def test_strings_and_empty_columns(self):
    data = {"text": np.array([b"a", b"b", b"c"], dtype=object),
            "empty": np.empty((3, 0, 2), np.float32)}
    ref = sw.create_reference_step(data)
    config = sw.create_config({k: v[-2:] for k, v in ref.items()}, "unused")
    dataset = replay.PatternDataset.from_tensor_slices(
        data, [config], True, lambda block: np.zeros(len(block["text"]), bool)).batch(2)
    with iter(dataset) as iterator:
      batch = next(iterator)
    np.testing.assert_array_equal(batch["text"], [[b"a", b"b"], [b"b", b"c"]])
    self.assertEqual(batch["empty"].shape, (2, 2, 0, 2))

  def test_cancel_repeated_input_without_matching_outputs(self):
    dataset = replay.PatternDataset.from_tensor_slices(
        self.data(), self.configs([sw.Condition.step_index() > 1000]), True,
        lambda block: block["is_last"], repeat=True).batch(2)
    iterator = iter(replay.prefetch(dataset))
    closer = threading.Thread(target=iterator.close, daemon=True)
    closer.start()
    closer.join(5)
    self.assertFalse(closer.is_alive())
    self.assertFalse(iterator._thread.is_alive())


if __name__ == "__main__":
  unittest.main()
