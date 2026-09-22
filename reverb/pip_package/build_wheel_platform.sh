#!/usr/bin/env bash
set -euo pipefail

archive=$1
output=$2
python_version=$3
revision=$4
jobs=$5

mkdir work
cd work
tar -xf "$archive"
source_dir=$PWD

flags=(
  "--jobs=$jobs"
  "--@rules_python//python/config_settings:python_version=$python_version"
  --repo_env=WHEEL_NAME=dm_reverb
  --repo_env=ML_WHEEL_TYPE=release
  "--repo_env=WHEEL_LOCAL_VERSION=g$revision"
)

bazel test "${flags[@]}" \
  //reverb:pybind_test \
  //reverb:trajectory_writer_test \
  //reverb:replay_dataset_test \
  //reverb:jax_iterator_test \
  //reverb:grain_test \
  //reverb/pip_package:build_wheel_test
bazel build "${flags[@]}" //reverb/pip_package:wheel
wheel=$(bazel cquery "${flags[@]}" --output=files //reverb/pip_package:wheel)

mkdir -p "$output"
case "$(uname -s):$(uname -m)" in
  Darwin:arm64)
    MACOSX_DEPLOYMENT_TARGET=12.0 uvx --from delocate delocate-wheel \
      --require-archs arm64 \
      --wheel-dir "$output" "$wheel"
    ;;
  Linux:x86_64|Linux:aarch64)
    uvx --with patchelf auditwheel repair \
      --plat "manylinux_2_39_$(uname -m)" \
      --wheel-dir "$output" "$wheel"
    ;;
  *) echo "Unsupported native build host" >&2; exit 1 ;;
esac

shopt -s nullglob
wheels=("$output"/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "Expected exactly one repaired wheel" >&2
  exit 1
fi

uv venv --python "$python_version" "$source_dir/.venv_test"
python="$source_dir/.venv_test/bin/python"
uv pip install --python "$python" --require-hashes \
  -r third_party/bzlmod/requirements.txt
uv pip install --python "$python" --no-deps "${wheels[0]}"
uv pip check --python "$python"

# The installed wheel must resolve imports without the source checkout.
cd ..
export JAX_PLATFORMS=cpu
"$python" -I - "$revision" <<'PY'
import importlib.metadata
import sys
import reverb
from reverb import grain

version = importlib.metadata.version("dm-reverb")
assert version.endswith("+g" + sys.argv[1]), version
print("Installed wheel:", version)
PY
for test in pybind trajectory_writer replay_dataset jax_iterator grain; do
  "$python" -I "$source_dir/reverb/${test}_test.py"
done
