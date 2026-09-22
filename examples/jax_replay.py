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

"""Run host-prefetched, sharded, or callback-driven JAX training."""

import argparse
import contextlib
import json

import input_pipeline
import jax
import jax.numpy as jnp
import numpy as np
import replay_fixture


def run(pattern='prefetch', steps=100, batch_size=64, length=32,
        width=128, depth=2, transport='grpc'):
  if min(steps, batch_size, length, width, depth) < 1:
    raise ValueError('Steps, dimensions, and prefetch depth must be positive')
  if jax.process_count() != 1:
    raise ValueError('This example supports a single JAX process')
  spec = jax.ShapeDtypeStruct((batch_size, length, width), np.float32)
  sharding = (input_pipeline.local_sharding(batch_size, 3)
              if pattern == 'sharded' else jax.local_devices()[0])
  with replay_fixture.replay_server(
      replay_fixture.Fixture(length, width), transport) as address:
    source = replay_fixture.ReplayBatches(
        address, batch_size, max_samples=steps * batch_size)
    replicated = (jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec())
                  if pattern == 'sharded' else sharding)
    params = jax.device_put(np.zeros((width,), np.float32), replicated)
    with contextlib.ExitStack() as stack:
      stack.callback(source.close)
      if pattern == 'callback':
        batches = stack.enter_context(input_pipeline.Prefetch(source, depth))
        train = input_pipeline.compile_scan(lambda: next(batches), spec, steps)
        try:
          params, losses = train(params)
          jax.block_until_ready((params, losses))
          loss = losses[-1]
        finally:
          jax.effects_barrier()
      else:
        step = input_pipeline.compile_step(
            spec, sharding if pattern == 'sharded' else None)
        batches = stack.enter_context(input_pipeline.Prefetch(
            source, depth, lambda data: jax.device_put(data, sharding)))
        for index in range(steps):
          params, loss = step(params, next(batches))
          if (index + 1) % depth == 0:
            jax.block_until_ready((params, loss))
        jax.block_until_ready((params, loss))
  return dict(pattern=pattern, steps=steps, loss=float(loss),
              parameter_norm=float(jnp.linalg.norm(params)),
              devices=[str(device) for device in jax.local_devices()])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pattern', choices=['prefetch', 'sharded', 'callback'],
                      default='prefetch')
  parser.add_argument('--steps', type=int, default=100)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--length', type=int, default=32)
  parser.add_argument('--width', type=int, default=128)
  parser.add_argument('--depth', type=int, default=2)
  parser.add_argument('--transport', choices=['local', 'grpc'], default='grpc')
  print(json.dumps(run(**vars(parser.parse_args())), sort_keys=True))


if __name__ == '__main__':
  main()
