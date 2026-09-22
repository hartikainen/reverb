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

"""Matching replay fixtures for TensorFlow and NumPy consumers."""

import contextlib
import dataclasses
import multiprocessing

import numpy as np
import reverb


@dataclasses.dataclass(frozen=True)
class Fixture:
  length: int = 32
  width: int = 128
  items: int = 128
  seed: int = 42


@contextlib.contextmanager
def local_server(config):
  server = reverb.Server([reverb.Table(
      'bench', reverb.selectors.Uniform(), reverb.selectors.Fifo(),
      config.items, reverb.rate_limiters.MinSize(1))])
  try:
    address = f'localhost:{server.port}'
    client = reverb.Client(address)
    rng = np.random.default_rng(config.seed)
    with client.trajectory_writer(num_keep_alive_refs=config.length) as writer:
      for _ in range(config.items):
        for _ in range(config.length):
          writer.append([rng.standard_normal(config.width).astype(np.float32)])
        writer.create_item('bench', 1.,
                           trajectory=[writer.history[0][-config.length:]])
      writer.flush()
    if client.server_info()['bench'].current_size != config.items:
      raise RuntimeError('Replay fixture insertion did not complete')
    yield address
  finally:
    server.stop()


def _serve(config, pipe):
  try:
    with local_server(config) as address:
      pipe.send((address, None))
      pipe.recv()
  except Exception as error:
    try:
      pipe.send((None, repr(error)))
    except (BrokenPipeError, EOFError):
      pass
  finally:
    pipe.close()


@contextlib.contextmanager
def replay_server(config, transport='grpc'):
  if transport == 'local':
    with local_server(config) as address:
      yield address
    return
  context = multiprocessing.get_context('spawn')
  parent, child = context.Pipe()
  process = context.Process(target=_serve, args=(config, child))
  process.start()
  child.close()
  try:
    if not parent.poll(120):
      raise TimeoutError('Replay server startup timed out')
    address, error = parent.recv()
    if error:
      raise RuntimeError(error)
    yield address
  finally:
    try:
      parent.send('stop')
    except (BrokenPipeError, EOFError):
      pass
    parent.close()
    process.join(timeout=10)
    if process.is_alive():
      process.terminate()
      process.join(timeout=10)


class ReplayBatches:
  """Project replay samples to data for a learner without priority updates."""

  def __init__(self, address, batch_size, workers=1, in_flight=128,
               max_samples=None):
    self._iterator = iter(reverb.ReplayDataset(
        address, 'bench', batch_size=batch_size, max_samples=max_samples,
        num_workers=workers, prefetch_size=in_flight,
        rate_limiter_timeout_ms=30000))

  def __iter__(self):
    return self

  def __next__(self):
    sample = next(self._iterator)
    return sample.data[0]

  def close(self):
    self._iterator.close()
