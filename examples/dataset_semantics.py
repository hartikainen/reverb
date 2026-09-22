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

"""Compare dataset values, metadata, boundaries, and remainders across runtimes."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import reverb
from reverb import structured_writer as sw
import tree


def encode(value):
  if isinstance(value, dict):
    return {key: encode(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [encode(item) for item in value]
  array = np.asarray(value)
  data = array.tolist()
  if array.dtype.kind in "OSU":
    data = tree.map_structure(
        lambda x: x.hex() if isinstance(x, bytes) else str(x), data)
  return dict(dtype=str(array.dtype), shape=list(array.shape), data=data)


def collect(dataset, backend, *, metadata=False):
  iterator = dataset.as_numpy_iterator() if backend == "tf" else iter(dataset)
  result = []
  keys = {}
  try:
    for sample in iterator:
      if metadata:
        normalized = np.array([
            keys.setdefault(int(key), len(keys))
            for key in np.asarray(sample.info.key).reshape(-1)], np.uint64)
        normalized = normalized.reshape(np.shape(sample.info.key))
        sample = sample._replace(info=sample.info._replace(key=normalized))
      result.append(encode(sample))
  finally:
    if backend != "tf":
      iterator.close()
  return result


def pattern_cases(backend):
  if backend == "tf":
    import tensorflow as tf
  else:
    from reverb import grain as rg
  data = {"x": np.arange(36, dtype=np.int32).reshape(12, 3),
          "last": np.isin(np.arange(12), [4, 8]),
          "keep": np.arange(12) % 3 == 0}
  ref = sw.create_reference_step(data)
  conditions = [[], [sw.Condition.is_end_episode()],
                [sw.Condition.step_index() % 2 == 0],
                [sw.Condition.steps_since_applied() >= 3],
                [sw.Condition.data(data)["keep"] == 1]]
  results = {}
  for respect in [False, True]:
    for index, condition in enumerate(conditions):
      for multiple in [False, True]:
        configs = [sw.create_config(
            {"window": ref["x"][-3::2], "last": ref["x"][-1]},
            "unused", condition)]
        if multiple:
          configs.append(sw.create_config(
              {"window": ref["x"][-2:], "last": ref["x"][-1]},
              "unused", condition))
        if backend == "tf":
          dataset = reverb.PatternDataset(
              tf.data.Dataset.from_tensor_slices(data), configs, respect,
              lambda step: step["last"])
        else:
          dataset = rg.PatternDataset.from_tensor_slices(
              data, configs, respect, lambda step: step["last"], input_block_size=5)
        name = f"pattern/{respect}/{index}/{multiple}"
        results[name] = collect(dataset.batch(4, drop_remainder=False), backend)
  return results


def sample_cases(backend):
  if backend == "tf":
    import tensorflow as tf
  else:
    from reverb import grain as rg
  results = {}
  for kind in ["trajectory", "timestep"]:
    for batch in [None, 4]:
      for drop in [False, True]:
        for timeout in [False, True]:
          server = reverb.Server([reverb.Table(
              "data", reverb.selectors.Fifo(), reverb.selectors.Fifo(), 8,
              reverb.rate_limiters.MinSize(1), max_times_sampled=1)])
          try:
            address = f"localhost:{server.port}"
            client = reverb.Client(address)
            with client.trajectory_writer(num_keep_alive_refs=3) as writer:
              for start in [0, 3]:
                for value in range(start, start + 3):
                  writer.append([np.array([value, value + 1], np.int32)])
                writer.create_item("data", 2.5, [writer.history[0][-3:]])
              writer.flush()
            limit = -1 if timeout else 2
            kwargs = dict(max_samples=limit, rate_limiter_timeout_ms=50)
            shape = (3, 2) if kind == "trajectory" else (2,)
            if backend == "tf":
              cls = reverb.TrajectoryDataset if kind == "trajectory" else reverb.TimestepDataset
              dataset = cls(address, "data", [tf.int32], [tf.TensorShape(shape)],
                            max_in_flight_samples_per_worker=2,
                            num_workers_per_iterator=1, max_samples_per_stream=1, **kwargs)
            else:
              cls = rg.TrajectoryDataset if kind == "trajectory" else rg.TimestepDataset
              dataset = cls(address, "data", dtypes=[np.int32], shapes=[shape],
                            prefetch_size=2, num_workers=1, max_samples_per_stream=1, **kwargs)
            if batch:
              dataset = dataset.batch(batch, drop_remainder=drop)
            name = f"{kind}/{batch}/{drop}/{timeout}"
            results[name] = collect(dataset, backend, metadata=True)
            del dataset
          finally:
            server.stop()
  return results


def run(backend):
  if backend == "tf":
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(2)
  try:
    version = importlib.metadata.version("dm-reverb")
  except importlib.metadata.PackageNotFoundError:
    version = "source"
  return dict(backend=backend, reverb=version,
              cases=dict(pattern_cases(backend), **sample_cases(backend)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=["tf", "grain"], required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  Path(args.output).write_text(json.dumps(run(args.backend), indent=2) + "\n")


if __name__ == "__main__":
  main()
