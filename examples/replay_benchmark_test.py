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

"""Check benchmark execution and reject invalid parity comparisons."""

import copy
import unittest

import compare_benchmarks
import replay_benchmark


class BenchmarkTest(unittest.TestCase):
  def args(self, *extra):
    return replay_benchmark.parse_args([
        '--revision=test-fixture', '--transport=local', '--length=2',
        '--width=3', '--items=4', '--batch-size=4', '--warmup=2',
        '--batches=4', '--sync-every=2', '--trials=1', *extra])

  def test_numpy_consumers(self):
    for consumer in ['host', 'device', 'learner', 'prefetch', 'callback']:
      with self.subTest(consumer=consumer):
        report = replay_benchmark.run(self.args('--consumer=' + consumer))
        self.assertEqual(report['status'], 'measured')
        self.assertGreater(report['median_elements_per_second'], 0)
        self.assertEqual(report['trials'][0]['batches'], 4)
        self.assertTrue(report['warnings'])

  def test_grain_adapters(self):
    for adapter in ['grain-trajectory', 'grain-timestep', 'grain-pattern', 'grain-step-pattern']:
      with self.subTest(adapter=adapter):
        report = replay_benchmark.run(self.args('--adapter=' + adapter))
        self.assertEqual(report['status'], 'measured')
        self.assertGreater(report['median_elements_per_second'], 0)

  def test_prefetch_without_separate_host_queue(self):
    report = replay_benchmark.run(self.args(
        '--consumer=prefetch', '--host-prefetch-batches=0'))
    self.assertEqual(report['status'], 'measured')
    self.assertEqual(report['config']['host_prefetch_batches'], 0)

  def test_grpc_fixture(self):
    report = replay_benchmark.run(self.args('--transport=grpc'))
    self.assertEqual(report['status'], 'measured')

  def test_missing_adapter_cannot_pass_comparison(self):
    baseline = replay_benchmark.run(self.args('--adapter=numpy-pattern'))
    result = compare_benchmarks.compare(baseline, baseline)
    self.assertEqual(result['status'], 'inconclusive')
    self.assertNotIn('median_elements_per_second', baseline)

  def test_comparison_rejects_noise_and_mismatched_workloads(self):
    baseline = dict(status='measured', config=dict(consumer='host', revision='a',
                                                  batch_size=4),
                    schema_version=1, benchmark_sha256='fixture',
                    semantics='sampled_trajectories', platform='fixture',
                    python='fixture', machine='fixture', cpu_count=1,
                    hostname='fixture', processor='fixture',
                    trials=[dict(elements_per_second=100)] * 3,
                    warnings=[], packages=dict(numpy='same'),
                    median_elements_per_second=100)
    candidate = copy.deepcopy(baseline)
    candidate['median_elements_per_second'] = 98
    candidate['trials'] = [dict(elements_per_second=98)] * 3
    self.assertEqual(compare_benchmarks.compare(baseline, candidate)['status'], 'pass')
    candidate['trials'] = [dict(elements_per_second=rate) for rate in [93, 98, 98]]
    self.assertEqual(compare_benchmarks.compare(baseline, candidate)['status'], 'inconclusive')
    candidate['median_elements_per_second'] = 80
    candidate['trials'] = [dict(elements_per_second=80)] * 3
    self.assertEqual(compare_benchmarks.compare(baseline, candidate)['status'], 'regression')
    candidate['config']['batch_size'] = 8
    self.assertEqual(compare_benchmarks.compare(baseline, candidate)['status'], 'inconclusive')
    candidate = copy.deepcopy(baseline)
    candidate['warnings'] = ['unstable timings']
    self.assertEqual(compare_benchmarks.compare(baseline, candidate)['status'], 'inconclusive')


if __name__ == '__main__':
  unittest.main()
