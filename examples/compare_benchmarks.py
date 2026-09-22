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

"""Compare matching replay measurements without hiding missing coverage."""

import argparse
import json
import math
from pathlib import Path
import statistics


def compare(baseline, candidate, max_regression=0.05):
  reasons = []
  if not 0 <= max_regression < 1:
    raise ValueError('`max_regression` must be in `[0, 1)`')
  for name, report in [('baseline', baseline), ('candidate', candidate)]:
    required = ['schema_version', 'benchmark_sha256', 'semantics', 'platform',
                'python', 'machine', 'hostname', 'processor', 'cpu_count']
    if any(key not in report for key in required):
      reasons.append(f'{name}: incomplete measurement metadata')
    if report.get('schema_version') != 1:
      reasons.append(f'{name}: unsupported measurement schema')
    if report.get('status') != 'measured':
      reasons.append(f'{name}: {report.get("reason", "measurement is unavailable")}')
    reasons.extend(f'{name}: {warning}' for warning in report.get('warnings', []))
    if len(report.get('trials', [])) < 3:
      reasons.append(f'{name}: insufficient repeated trials')
    for trial in report.get('trials', []):
      rate = trial.get('elements_per_second', 0)
      if not math.isfinite(rate) or rate <= 0:
        reasons.append(f'{name}: invalid trial rate')
  for key in ['schema_version', 'benchmark_sha256', 'semantics', 'platform',
              'python', 'machine', 'hostname', 'processor', 'cpu_count']:
    if baseline.get(key) != candidate.get(key):
      reasons.append(f'Mismatched `{key}`')
  ignored = {'adapter', 'revision', 'output'}
  settings = lambda report: {key: value for key, value in report['config'].items()
                             if key not in ignored}
  if settings(baseline) != settings(candidate):
    reasons.append('Mismatched workload or consumer settings')
  packages = ['numpy']
  if baseline['config']['consumer'] != 'host':
    packages += ['jax', 'jaxlib']
  for package in packages:
    if baseline['packages'].get(package) != candidate['packages'].get(package):
      reasons.append(f'Mismatched `{package}` version')
  if baseline['config']['consumer'] != 'host':
    device_sets = lambda report: {tuple(trial.get('devices', []))
                                  for trial in report.get('trials', [])}
    if device_sets(baseline) != device_sets(candidate):
      reasons.append('Mismatched learner devices')
  if reasons:
    return dict(status='inconclusive', reasons=reasons)
  baseline_rates = [trial['elements_per_second'] for trial in baseline['trials']]
  candidate_rates = [trial['elements_per_second'] for trial in candidate['trials']]
  ratio = statistics.median(candidate_rates) / statistics.median(baseline_rates)
  lower = min(candidate_rates) / max(baseline_rates)
  upper = max(candidate_rates) / min(baseline_rates)
  status = ('pass' if lower >= 1 - max_regression else
            'regression' if upper < 1 - max_regression else 'inconclusive')
  return dict(status=status, observed_ratio_range=[lower, upper],
              candidate_over_baseline=ratio, max_regression=max_regression,
              baseline_revision=baseline['config']['revision'],
              candidate_revision=candidate['config']['revision'])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', required=True)
  parser.add_argument('--candidate', required=True)
  parser.add_argument('--max-regression', type=float, default=0.05)
  args = parser.parse_args()
  result = compare(json.loads(Path(args.baseline).read_text()),
                   json.loads(Path(args.candidate).read_text()),
                   args.max_regression)
  print(json.dumps(result, indent=2, sort_keys=True))
  raise SystemExit({'pass': 0, 'regression': 1, 'inconclusive': 2}[result['status']])


if __name__ == '__main__':
  main()
