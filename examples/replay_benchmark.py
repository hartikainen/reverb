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

"""Measure replay adapters and JAX consumers with matched fixtures."""

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time

import numpy as np
import replay_fixture
import reverb


ADAPTERS = ['numpy-trajectory', 'numpy-timestep', 'numpy-pattern',
            'tf-trajectory', 'tf-timestep', 'tf-pattern',
            'grain-trajectory', 'grain-timestep', 'grain-pattern',
            'grain-step-pattern', 'tf-generator-pattern']


class UnsupportedAdapter(Exception):
  pass


def check_adapter(adapter):
  if adapter.startswith('grain-'):
    try:
      from reverb import grain as replay_grain
    except ImportError as error:
      raise UnsupportedAdapter('Install the Reverb `grain` extra') from error
    return
  names = {'numpy-trajectory': 'ReplayDataset',
           'tf-trajectory': 'TrajectoryDataset',
           'tf-timestep': 'TimestepDataset', 'tf-pattern': 'PatternDataset',
           'tf-generator-pattern': 'PatternDataset'}
  if adapter not in names:
    raise UnsupportedAdapter(f'No implemented benchmark adapter for `{adapter}`')
  if not hasattr(reverb, names[adapter]):
    raise UnsupportedAdapter(f'Installed Reverb does not expose `{names[adapter]}`')


class GrainBatches:
  def __init__(self, args, address):
    import grain
    from reverb import grain as replay_grain
    if args.adapter.endswith('pattern'):
      from reverb import structured_writer
      rng = np.random.default_rng(args.seed)
      data = rng.standard_normal((args.episode_length, args.width)).astype(np.float32)
      inputs = {'data': data,
                'is_last': np.arange(args.episode_length) == args.episode_length - 1}
      reference = structured_writer.create_reference_step({'data': 0, 'is_last': 0})
      config = structured_writer.create_config(
          {'data': reference['data'][-args.length:]}, 'unused')
      if args.adapter == 'grain-step-pattern':
        steps = [{key: value[i] for key, value in inputs.items()}
                 for i in range(args.episode_length)]
        inputs = grain.MapDataset.source(steps).repeat().to_iter_dataset(
            grain.ReadOptions(num_threads=0, prefetch_buffer_size=0))
        dataset = replay_grain.PatternDataset(
            inputs, [config], True, lambda step: step['is_last'])
      else:
        dataset = replay_grain.PatternDataset.from_tensor_slices(
            inputs, [config], True, lambda step: step['is_last'], repeat=True)
      dataset = dataset.batch(args.batch_size, drop_remainder=True)
    else:
      cls = (replay_grain.TrajectoryDataset if args.adapter == 'grain-trajectory'
             else replay_grain.TimestepDataset)
      dataset = cls(address, 'bench', args.batch_size,
                    prefetch_size=args.in_flight, num_workers=args.workers,
                    rate_limiter_timeout_ms=30000)
    self._iterator = iter(dataset)
    self._pattern = args.adapter.endswith('pattern')

  def __next__(self):
    sample = next(self._iterator)
    return sample['data'] if self._pattern else sample.data[0]

  def close(self):
    self._iterator.close()


class TensorFlowBatches:
  def __init__(self, args, address):
    import tensorflow as tf
    if args.adapter.endswith('pattern'):
      from reverb import structured_writer
      rng = np.random.default_rng(args.seed)
      steps = args.episode_length
      arrays = {
          'data': rng.standard_normal((steps, args.width)).astype(np.float32),
          'is_last': np.arange(steps) == steps - 1,
      }
      if args.adapter == 'tf-generator-pattern':
        records = [{key: value[i] for key, value in arrays.items()} for i in range(steps)]
        def generate():
          while True:
            yield from records
        inputs = tf.data.Dataset.from_generator(generate, output_signature={
            'data': tf.TensorSpec((args.width,), tf.float32),
            'is_last': tf.TensorSpec((), tf.bool)})
      else:
        inputs = tf.data.Dataset.from_tensor_slices(arrays).repeat()
      reference = structured_writer.create_reference_step(inputs.element_spec)
      config = structured_writer.create_config(
          {'data': reference['data'][-args.length:]}, 'unused')
      dataset = reverb.PatternDataset(
          inputs, [config], respect_episode_boundaries=True,
          is_end_of_episode=lambda step: step['is_last'])
    else:
      cls = (reverb.TrajectoryDataset if args.adapter == 'tf-trajectory'
             else reverb.TimestepDataset)
      shape = ([args.length, args.width] if args.adapter == 'tf-trajectory'
               else [args.width])
      dataset = cls(address, 'bench', [tf.float32], [tf.TensorShape(shape)],
                    max_in_flight_samples_per_worker=args.in_flight,
                    num_workers_per_iterator=args.workers,
                    rate_limiter_timeout_ms=30000)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = args.tf_threads
    options.threading.max_intra_op_parallelism = 1
    self._dataset = dataset.with_options(options)
    self._iterator = self._dataset.as_numpy_iterator()
    self._pattern = args.adapter.endswith('pattern')

  def __next__(self):
    sample = next(self._iterator)
    return sample['data'] if self._pattern else sample.data[0]

  def close(self):
    self._iterator = None
    self._dataset = None


def metadata(args):
  packages = {}
  for name in ['dm-reverb', 'numpy', 'jax', 'jaxlib', 'tensorflow', 'grain']:
    try:
      packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      pass
  digest = hashlib.sha256()
  for filename in ['replay_benchmark.py', 'replay_fixture.py', 'input_pipeline.py']:
    digest.update(Path(__file__).with_name(filename).read_bytes())
  return dict(schema_version=1, benchmark_sha256=digest.hexdigest(),
              config=vars(args), packages=packages,
              python=platform.python_version(), platform=platform.platform(),
              hostname=platform.node(), processor=platform.processor(),
              machine=platform.machine(), cpu_count=os.cpu_count(),
              reverb_path=reverb.__file__)


def measure(args, address):
  shape = ((args.batch_size, args.width) if args.adapter.endswith('timestep')
           else (args.batch_size, args.length, args.width))
  with contextlib.ExitStack() as stack:
    if args.adapter == 'numpy-trajectory':
      source = replay_fixture.ReplayBatches(
          address, args.batch_size, args.workers, args.in_flight)
    elif args.adapter.startswith('grain-'):
      source = GrainBatches(args, address)
    else:
      source = TensorFlowBatches(args, address)
    stack.callback(source.close)
    probe = next(source)
    if probe.shape != shape or probe.dtype != np.float32:
      raise ValueError(f'Unexpected batch specification: {probe.shape}, {probe.dtype}')
    read_latencies = []
    params = result = None
    if args.consumer != 'host':
      import input_pipeline
      import jax
      spec = jax.ShapeDtypeStruct(shape, np.float32)
      params = jax.device_put(np.zeros(args.width, np.float32))
      if args.consumer in ('learner', 'prefetch'):
        step = input_pipeline.compile_step(spec)
      jax.block_until_ready(params)
      if args.consumer in ('prefetch', 'callback'):
        transform = (jax.device_put if args.consumer == 'prefetch' else lambda x: x)
        source = stack.enter_context(input_pipeline.Prefetch(
            source, args.prefetch_batches, transform))

    def read():
      start = time.perf_counter()
      data = next(source)
      read_latencies.append(time.perf_counter() - start)
      return data

    if args.consumer == 'callback':
      scan = input_pipeline.compile_scan(read, spec, args.sync_every)

    def consume(count):
      nonlocal params, result
      if args.consumer == 'callback':
        for _ in range(count // args.sync_every):
          params, result = scan(params)
          jax.block_until_ready((params, result))
      else:
        pending_transfers = []
        for index in range(count):
          data = read()
          if args.consumer == 'device':
            result = jax.device_put(data)
            pending_transfers.append(result)
          elif args.consumer in ('learner', 'prefetch'):
            params, result = step(params, jax.device_put(data))
          if args.consumer != 'host' and (index + 1) % args.sync_every == 0:
            jax.block_until_ready((params, result, pending_transfers))
            pending_transfers.clear()
        if args.consumer != 'host':
          jax.block_until_ready((params, result, pending_transfers))

    try:
      consume(args.warmup)
      read_latencies.clear()
      load_start = os.getloadavg()
      cpu_start = time.process_time()
      start = time.perf_counter()
      consume(args.batches)
      elapsed = time.perf_counter() - start
      cpu_seconds = time.process_time() - cpu_start
    finally:
      if args.consumer != 'host':
        jax.effects_barrier()
    elements = args.batches * args.batch_size
    timesteps = elements * (1 if args.adapter.endswith('timestep') else args.length)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
      peak_rss *= 1024
    result = dict(seconds=elapsed, batches=args.batches,
                  batches_per_second=args.batches / elapsed,
                  elements_per_second=elements / elapsed,
                  output_timesteps_per_second=timesteps / elapsed,
                  data_bytes_per_second=args.batches * np.prod(shape).item() * 4 / elapsed,
                  read_wait_p50_seconds=float(np.percentile(read_latencies, 50)),
                  read_wait_p95_seconds=float(np.percentile(read_latencies, 95)),
                  client_cpu_seconds=cpu_seconds, client_peak_rss_bytes=peak_rss,
                  load_average_start=load_start, load_average_end=os.getloadavg())
    if args.consumer != 'host':
      result['devices'] = [f'{device.platform}:{device.id}:{device.device_kind}'
                           for device in jax.local_devices()]
    return result


def run(args):
  report = metadata(args)
  try:
    check_adapter(args.adapter)
  except UnsupportedAdapter as error:
    return dict(report, status='unsupported', reason=str(error))
  if args.adapter.startswith('tf-'):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(args.tf_threads)
    tf.config.threading.set_intra_op_parallelism_threads(1)
  config = replay_fixture.Fixture(args.length, args.width, args.items, args.seed)
  server = (contextlib.nullcontext(None) if args.adapter.endswith('pattern')
            else replay_fixture.replay_server(config, args.transport))
  with server as address:
    trials = []
    for _ in range(args.trials):
      trials.append(measure(args, address))
      gc.collect()
  rates = [trial['elements_per_second'] for trial in trials]
  relative_stddev = statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else None
  warnings = []
  if min(trial['seconds'] for trial in trials) < args.min_seconds:
    warnings.append('Timing windows are short; increase `--batches`')
  if relative_stddev is not None and relative_stddev > 0.1:
    warnings.append('Trial variation exceeds the benchmark noise threshold')
  peak_load = max(trial[key][0] for trial in trials
                  for key in ['load_average_start', 'load_average_end'])
  if peak_load > (os.cpu_count() or 1):
    warnings.append('Host load exceeds the logical CPU count')
  return dict(report, status='measured', trials=trials, warnings=warnings,
              median_elements_per_second=statistics.median(rates),
              relative_stddev=relative_stddev,
              semantics=('local_episode_windows' if args.adapter.endswith('pattern')
                         else 'sampled_items_then_timesteps' if args.adapter.endswith('timestep')
                         else 'sampled_trajectories'))


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--adapter', choices=ADAPTERS, default='numpy-trajectory')
  parser.add_argument('--consumer', choices=['host', 'device', 'learner', 'prefetch', 'callback'], default='host')
  parser.add_argument('--revision', required=True, help='Commit used to build the installed Reverb')
  parser.add_argument('--transport', choices=['local', 'grpc'], default='grpc')
  for name, default in [('batch-size', 64), ('length', 32), ('width', 128),
                        ('items', 128), ('workers', 1), ('in-flight', 128),
                        ('prefetch-batches', 2), ('batches', 4096), ('warmup', 128),
                        ('trials', 5), ('sync-every', 16), ('tf-threads', 2),
                        ('episode-length', 128)]:
    parser.add_argument('--' + name, type=int, default=default)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--min-seconds', type=float, default=2.)
  parser.add_argument('--output', default=None)
  args = parser.parse_args(argv)
  for name, value in vars(args).items():
    if isinstance(value, int) and name != 'seed' and value < 1:
      parser.error(f'`{name}` must be positive')
  if args.min_seconds <= 0:
    parser.error('`min_seconds` must be positive')
  if args.episode_length < args.length:
    parser.error('`episode_length` must cover `length`')
  if args.consumer == 'callback' and (
      args.batches % args.sync_every or args.warmup % args.sync_every):
    parser.error('Callback batch counts must be multiples of `sync_every`')
  return args


def main():
  args = parse_args()
  report = run(args)
  text = json.dumps(report, indent=2, sort_keys=True) + '\n'
  if args.output:
    Path(args.output).write_text(text)
  else:
    print(text, end='')
  if report['status'] == 'unsupported':
    raise SystemExit(2)


if __name__ == '__main__':
  main()
