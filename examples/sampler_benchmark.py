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

"""Measure host sampling throughput and client peak RSS over loopback gRPC.

Run each trial in a separate process to isolate allocator high-water marks.
Payloads contain seeded random bytes, and insertion finishes before sampling.
The server runs in a separate process so sampling uses gRPC. This benchmark
excludes producer throughput, device transfers, and learner computation.
"""

import argparse
import contextlib
import importlib.metadata
import json
import multiprocessing
import platform
import resource
import sys
import time

import numpy as np
import reverb


def _rss_mib():
  scale = 1024**2 if sys.platform == 'darwin' else 1024
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def _payload(config, item):
  rng = np.random.default_rng(config.seed + item)
  return [rng.integers(0, 256, (config.length, config.width), dtype=np.uint8)
          for _ in range(config.columns)]


def _serve(config, pipe):
  server = None
  try:
    server = reverb.Server([reverb.Table(
        'bench', reverb.selectors.Uniform(), reverb.selectors.Fifo(),
        config.items, reverb.rate_limiters.MinSize(1))])
    address = f'localhost:{server.port}'
    client = reverb.Client(address)
    with client.trajectory_writer(num_keep_alive_refs=config.length) as writer:
      for column in range(config.columns):
        writer.configure((column,), num_keep_alive_refs=config.length,
                         max_chunk_length=config.chunk_length)
      for item in range(config.items):
        values = _payload(config, item)
        for step in range(config.length):
          writer.append([column[step] for column in values])
        writer.create_item('bench', item + 1., trajectory=[
            writer.history[column][-config.length:]
            for column in range(config.columns)])
      writer.flush()
    if client.server_info()['bench'].current_size != config.items:
      raise RuntimeError('Replay fixture insertion did not complete')
    pipe.send((address, None))
    pipe.recv()
  except Exception as error:
    with contextlib.suppress(BrokenPipeError, EOFError):
      pipe.send((None, repr(error)))
  finally:
    if server is not None:
      server.stop()
    pipe.close()


@contextlib.contextmanager
def _server(config):
  context = multiprocessing.get_context('spawn')
  parent, child = context.Pipe()
  process = context.Process(target=_serve, args=(config, child))
  process.start()
  child.close()
  try:
    if not parent.poll(180):
      raise TimeoutError('Replay fixture startup timed out')
    address, error = parent.recv()
    if error:
      raise RuntimeError(error)
    yield address
  finally:
    with contextlib.suppress(BrokenPipeError, EOFError):
      parent.send('stop')
    parent.close()
    process.join(timeout=15)
    if process.is_alive():
      process.terminate()
      process.join(timeout=15)
    if process.is_alive():
      process.kill()
      process.join()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--label', required=True)
  parser.add_argument('--length', type=int, default=32)
  parser.add_argument('--width', type=int, default=128)
  parser.add_argument('--columns', type=int, default=32)
  parser.add_argument('--chunk-length', type=int, default=32)
  parser.add_argument('--items', type=int, default=4)
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--prefetch', type=int, default=1)
  parser.add_argument('--warmup', type=int, default=8)
  parser.add_argument('--seconds', type=float, default=5)
  parser.add_argument('--seed', type=int, default=42)
  config = parser.parse_args()
  for name in ('length', 'width', 'columns', 'chunk_length', 'items', 'workers',
               'prefetch', 'warmup', 'seconds'):
    if getattr(config, name) <= 0:
      parser.error(f'`{name}` must be positive')
  if config.chunk_length > config.length:
    parser.error('`chunk_length` must not exceed `length`')
  sample_bytes = config.length * config.width * config.columns
  result = {'config': vars(config), 'sample_bytes': sample_bytes,
            'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__,
            'tensorflow': importlib.metadata.version('tensorflow')}
  with _server(config) as address:
    result['client_peak_before_sampling_mib'] = _rss_mib()
    dataset = reverb.ReplayDataset(
        address, 'bench', batch_size=1, num_workers=config.workers,
        prefetch_size=config.prefetch, rate_limiter_timeout_ms=30000)
    with iter(dataset) as iterator:
      for _ in range(config.warmup):
        sample = next(iterator)
      result['client_peak_after_warmup_mib'] = _rss_mib()
      count = 0
      start = time.perf_counter()
      elapsed = 0.
      while elapsed < config.seconds:
        sample = next(iterator)
        count += 1
        elapsed = time.perf_counter() - start
      result.update(samples=count, seconds=elapsed,
                    samples_per_second=count / elapsed,
                    payload_gib_per_second=count * sample_bytes / elapsed / 1024**3,
                    client_peak_after_sampling_mib=_rss_mib())
    item = int(sample.info.priority[0]) - 1
    expected = _payload(config, item)
    if len(sample.data) != config.columns:
      raise AssertionError('Sample column count does not match the fixture')
    for actual, wanted in zip(sample.data, expected):
      np.testing.assert_array_equal(actual[0], wanted)
  print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
  main()
