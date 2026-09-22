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

"""Check dataset semantics against TensorFlow-generated outputs."""

import json
from pathlib import Path
import unittest

import dataset_semantics


class DatasetSemanticsTest(unittest.TestCase):

  def test_matches_tensorflow(self):
    fixture = Path(__file__).with_name("testdata") / "tensorflow_dataset_semantics.json"
    expected = json.loads(fixture.read_text())["cases"]
    actual = dataset_semantics.run("grain")["cases"]
    self.assertEqual(actual.keys(), expected.keys())
    for name in expected:
      with self.subTest(case=name):
        self.assertEqual(actual[name], expected[name])


if __name__ == "__main__":
  unittest.main()
