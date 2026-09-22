#!/usr/bin/env python3
"""Build tested wheels from a Git revision on native macOS and Docker Linux."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from email.parser import BytesParser


PLATFORMS = {
    "macos_arm64": None,
    "linux_x86_64": "linux/amd64",
    "linux_arm64": "linux/arm64",
}


def run(args, *, capture=False, **kwargs):
  print("+ " + shlex.join(str(arg) for arg in args), flush=True)
  result = subprocess.run(
      args, check=True, text=True,
      stdout=subprocess.PIPE if capture else None, **kwargs)
  return result.stdout.strip() if capture else None


def defaults():
  targets = ["linux_arm64", "linux_x86_64"]
  if platform.system() == "Darwin" and platform.machine() == "arm64":
    targets.insert(0, "macos_arm64")
  return targets


def check_wheel(directory, target, python_version, revision):
  wheels = list(directory.glob("*.whl"))
  if len(wheels) != 1:
    raise ValueError("Expected exactly one wheel in " + str(directory))
  wheel = wheels[0]
  python_tag = "cp" + python_version.replace(".", "")
  tags = wheel.stem.rsplit("-", 3)[1:]
  expected_platform = {
      "macos_arm64": "macosx_12_0_arm64",
      "linux_x86_64": "manylinux_2_39_x86_64",
      "linux_arm64": "manylinux_2_39_aarch64",
  }[target]
  if (len(tags) != 3 or tags[:2] != [python_tag, python_tag]
      or expected_platform not in tags[2].split(".")):
    raise ValueError("Unexpected wheel tags: " + wheel.name)
  with zipfile.ZipFile(wheel) as archive:
    metadata_files = [name for name in archive.namelist()
                      if name.endswith(".dist-info/METADATA")]
    if len(metadata_files) != 1:
      raise ValueError("Expected one wheel metadata file")
    metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
  if metadata.get("Name", "").replace("_", "-") != "dm-reverb":
    raise ValueError("Unexpected wheel distribution")
  if not metadata.get("Version", "").endswith("+g" + revision):
    raise ValueError("Wheel version does not identify the source commit")
  return wheel, metadata["Version"]


def docker_build(docker, context, archive, output, target, python_version,
                 revision, jobs, bazel_version):
  architecture = PLATFORMS[target].split("/")[1]
  image = "reverb-wheel-builder:" + architecture
  run(docker + ["buildx", "build", "--load", "--platform", PLATFORMS[target],
                "--build-arg", "BAZEL_VERSION=" + bazel_version,
                "--tag", image, str(context)])
  container = "reverb-wheel-" + uuid.uuid4().hex
  run(docker + ["create", "--name", container, "--platform", PLATFORMS[target],
                "--mount", "type=volume,src=reverb-wheel-cache-" + architecture
                + ",dst=/home/reverb/.cache", image,
                "/tmp/source.tar", "/home/reverb/dist", python_version,
                revision, str(jobs)])
  try:
    # Copying inputs also works when the Docker daemon is on another machine.
    run(docker + ["cp", str(archive), container + ":/tmp/source.tar"])
    run(docker + ["start", "--attach", container])
    status = run(docker + ["inspect", "--format", "{{.State.ExitCode}}",
                          container], capture=True)
    if status != "0":
      raise RuntimeError("Wheel build exited with status " + status)
    run(docker + ["cp", container + ":/home/reverb/dist/.", str(output)])
  finally:
    subprocess.run(docker + ["rm", "--force", container], check=False)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--platforms", nargs="+", choices=PLATFORMS,
                      default=defaults())
  parser.add_argument("--revision", default="HEAD")
  parser.add_argument("--python", choices=["3.11", "3.12", "3.13"], default="3.13")
  parser.add_argument("--output-dir", type=Path, default=Path("dist"))
  parser.add_argument("--docker-context", help="Docker context for Linux builds")
  parser.add_argument("--jobs", type=int, default=4)
  args = parser.parse_args(argv)
  if args.jobs < 1:
    parser.error("`--jobs` must be positive")
  if len(set(args.platforms)) != len(args.platforms):
    parser.error("`--platforms` must not contain duplicates")
  if "macos_arm64" in args.platforms and (
      platform.system(), platform.machine()) != ("Darwin", "arm64"):
    parser.error("`macos_arm64` requires an Apple Silicon Mac")

  package = Path(__file__).resolve().parent
  repo = package.parent.parent
  revision = run(["git", "rev-parse", "--verify", "--end-of-options",
                  args.revision + "^{commit}"], cwd=repo, capture=True)
  config = run(["git", "show", revision + ":reverb/pip_package/wheel_config.bzl"],
               cwd=repo, capture=True)
  if "WHEEL_LOCAL_VERSION" not in config:
    parser.error("The selected revision must support `WHEEL_LOCAL_VERSION`")
  bazel_version = run(["git", "show", revision + ":.bazelversion"],
                      cwd=repo, capture=True)
  output = args.output_dir.resolve() / revision
  for target in args.platforms:
    if (output / target).exists():
      parser.error("Output already exists: " + str(output / target))

  docker = ["docker"]
  if args.docker_context:
    docker += ["--context", args.docker_context]
  if any(PLATFORMS[target] for target in args.platforms):
    run(docker + ["info"], capture=True)
  if "macos_arm64" in args.platforms:
    for tool in ("bazel", "uv", "xcrun"):
      if not shutil.which(tool):
        parser.error("Required macOS build tool is missing: " + tool)
    run(["xcrun", "--find", "clang"], capture=True)

  print("Building committed sources at " + revision, flush=True)
  print("Linux builders use Ubuntu 24.04; emulated builds may be slow.", flush=True)
  output.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="reverb-wheels-") as temporary:
    temporary = Path(temporary)
    archive = temporary / "source.tar"
    run(["git", "archive", "--format=tar", "--output=" + str(archive), revision],
        cwd=repo)
    context = temporary / "docker"
    context.mkdir()
    shutil.copyfile(repo / "docker/wheel.dockerfile", context / "Dockerfile")
    shutil.copyfile(package / "build_wheel_platform.sh", context / "build.sh")
    for target in args.platforms:
      with tempfile.TemporaryDirectory(prefix=".building-", dir=output) as staging:
        staging = Path(staging)
        if PLATFORMS[target]:
          docker_build(docker, context, archive, staging, target, args.python,
                       revision, args.jobs, bazel_version)
        else:
          native = temporary / "native"
          native.mkdir()
          run(["bash", str(package / "build_wheel_platform.sh"), str(archive),
               str(staging), args.python, revision, str(args.jobs)], cwd=native)
        wheel, version = check_wheel(staging, target, args.python, revision)
        manifest = {
            "revision": revision, "platform": target, "python": args.python,
            "version": version, "wheel": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
        (staging / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output / target)
        print("Tested wheel: " + str(output / target / wheel.name), flush=True)


if __name__ == "__main__":
  main()
