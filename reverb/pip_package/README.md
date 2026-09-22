# Building Reverb from source

Reverb uses Bzlmod for source targets, tests, and Python wheels. Install Bazelisk
(or the Bazel version in `.bazelversion`), a C++ compiler, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). macOS builds
require Xcode command-line tools.

`MODULE.bazel` registers Python `3.10`, `3.11`, `3.12`, and `3.13` toolchains,
with `3.13` as the default. Wheel targets support Linux `x86_64`, Linux
`aarch64`, and macOS Apple Silicon. They compile against TensorFlow `2.21.0`.

## Build platform wheels

From an Apple Silicon Mac with Python, Docker Desktop, Bazelisk, `uv`, and
Xcode command-line tools installed:

```sh
python3 reverb/pip_package/build_wheels.py
```

The launcher builds `macos_arm64` natively and `linux_arm64` and `linux_x86_64`
in Docker. On Linux, the default includes the Linux targets. Docker must support
executing each requested architecture. Docker Desktop supplies emulation on
Apple Silicon; emulated C++ compilation can be slower than using a native host.

Each build uses a Git archive of `HEAD`, excluding uncommitted changes. Select
another committed revision with `--revision`. That revision must include
`WHEEL_LOCAL_VERSION` support. Wheels carry `+g<commit>` in their package version
and are written to `dist/<commit>/<platform>/`, alongside `build.json` with the
source revision and wheel hash. Existing platform output directories are never
overwritten. Use `--output-dir` to keep a separate run.

The Linux builder uses Ubuntu `24.04` and repairs wheels for `manylinux_2_39`.
This matches Ubuntu `24.04` deployments but requires a different builder for
older glibc runtimes. macOS wheels target `macosx_12_0_arm64`.

Source tests and installed-wheel tests must pass before the launcher retains a
wheel. Installed-wheel tests run outside the source checkout, using Python
packages from `third_party/bzlmod/requirements.txt`. `--python` selects a supported
interpreter, and `--jobs` limits Bazel concurrency. Builds run sequentially.
Docker builder images and architecture-specific cache volumes persist for reuse;
the build containers are removed when each build completes or fails.

To build the Linux wheel on a native remote Docker host:

```sh
python3 reverb/pip_package/build_wheels.py \
  --platforms linux_x86_64 --docker-context amd64-builder
```

`amd64-builder` must be an existing Docker context. Inputs and outputs travel
through `docker cp`, so a remote host does not need a checkout or host bind mounts.
Use `--platforms macos_arm64` or `--platforms linux_arm64` for the other builds.
The launcher builds wheels without publishing them.

## Build wheels on the host

```sh
bash oss_build.sh --release --python '3.13'
```

`oss_build.sh` builds and tests the source targets, repairs wheel dependencies
with `auditwheel` or `delocate`, and installs the wheel for Python tests.
`--python '3.11 3.12'` selects multiple interpreters, `--output_dir` selects the
output directory (default `dist`), and `--python_tests false` skips the installed
wheel tests. Omitting `--release` produces a dated `dm_reverb_nightly` wheel.
Repairing wheels requires access to the package index for the repair tool.

To build a release wheel directly:

```sh
bazel build \
  --@rules_python//python/config_settings:python_version=3.13 \
  //reverb/pip_package:wheel
```

Bazel downloads hash-pinned build dependencies during repository resolution.
The wheel action uses the declared build backend without an isolated package
installation. It includes Reverb's transitive Python sources, generated schemas,
and shared libraries, excluding third-party packages.

For a nightly wheel:

```sh
bazel build \
  --@rules_python//python/config_settings:python_version=3.13 \
  --repo_env=WHEEL_NAME=dm_reverb_nightly \
  --repo_env=ML_WHEEL_TYPE=nightly \
  --repo_env=ML_WHEEL_BUILD_DATE="$(date '+%Y%m%d')" \
  //reverb/pip_package:wheel
```

Nightly metadata requests `tf_nightly~=2.21.0.dev`. The native build still uses
the TensorFlow release pinned by the module graph, so runtime compatibility with
a particular nightly distribution needs a wheel installation test.

## Update Python dependencies

Edit `third_party/bzlmod/requirements.in` and regenerate the universal lock:

```sh
uv pip compile --python-version 3.10 --universal \
  --generate-hashes --no-header --no-annotate \
  third_party/bzlmod/requirements.in \
  -o third_party/bzlmod/requirements.txt
bazel mod graph
```

TensorFlow's headers and shared library must agree with Reverb's native
Abseil, gRPC, and Protobuf dependencies. Changing TensorFlow requires updating
its Python lock entries, the schema archive in
`third_party/bzlmod/repositories.bzl`, the native constraints in `MODULE.bazel`,
and the metadata in `reverb_version.bzl`. The separate `protobuf` module supplies
Bazel rules, while `reverb_protobuf` supplies the compatible native generators.
