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

"""Feed Grain replay batches to a compiled, optionally sharded JAX learner."""

import argparse
import json

import input_pipeline
import jax
import numpy as np
import replay_fixture
from reverb import grain as replay_grain
from reverb import replay_sample


def run(steps=100, batch_size=64, length=32, width=128, depth=2,
        sharded=False):
  if min(steps, batch_size, length, width, depth) < 1:
    raise ValueError("Steps, dimensions, and queue depth must be positive")
  if jax.process_count() != 1:
    raise ValueError("This example supports a single JAX process")
  placement = (input_pipeline.local_sharding(batch_size, 3) if sharded
               else jax.local_devices()[0])
  replicated = (jax.sharding.NamedSharding(placement.mesh, jax.sharding.PartitionSpec())
                if sharded else placement)
  spec = jax.ShapeDtypeStruct((batch_size, length, width), np.float32)
  update = input_pipeline.compile_step(spec, placement if sharded else None)
  params = jax.device_put(np.zeros(width, np.float32), replicated)
  with replay_fixture.replay_server(replay_fixture.Fixture(length, width)) as address:
    source = replay_grain.TrajectoryDataset(
        address, 'bench', batch_size, max_samples=steps * batch_size)
    host = replay_grain.prefetch(source, depth)
    prepared = host.map(lambda sample: replay_sample.ReplaySample(
        sample.info, jax.device_put(sample.data[0], placement)))
    with iter(replay_grain.prefetch(prepared, depth)) as batches:
      try:
        for index, sample in enumerate(batches):
          params, loss = update(params, sample.data)
          # Host keys retain `uint64` precision for optional priority updates.
          assert sample.info.key.dtype == np.dtype('uint64')
          if (index + 1) % depth == 0:
            jax.block_until_ready((params, loss))
      finally:
        jax.block_until_ready(params)
  return dict(loss=float(loss), steps=steps, sharded=sharded)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name, default in [('steps', 100), ('batch-size', 64), ('length', 32),
                        ('width', 128), ('depth', 2)]:
    parser.add_argument('--' + name, type=int, default=default)
  parser.add_argument('--sharded', action='store_true')
  print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == '__main__':
  main()
