"""Check artifact validation and Docker failure handling without a compiler."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import build_wheels


REVISION = "a" * 40


def make_wheel(directory, *, version=None, tag="manylinux_2_39_aarch64",
               python_version="3.13"):
  version = version or "0.15.0+g" + REVISION
  python_tag = "cp" + python_version.replace(".", "")
  path = directory / f"dm_reverb-{version}-{python_tag}-{python_tag}-{tag}.whl"
  with zipfile.ZipFile(path, "w") as archive:
    archive.writestr(
        f"dm_reverb-{version}.dist-info/METADATA",
        f"Name: dm-reverb\nVersion: {version}\n")
  return path


class WheelValidationTest(unittest.TestCase):

  def test_matching_wheel(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      wheel = make_wheel(root)
      self.assertEqual(
          build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)[0],
          wheel)

  def test_rejects_wrong_architecture(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      make_wheel(root, tag="manylinux_2_39_x86_64")
      with self.assertRaisesRegex(ValueError, "tags"):
        build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)

  def test_rejects_wrong_python(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      make_wheel(root)
      with self.assertRaisesRegex(ValueError, "tags"):
        build_wheels.check_wheel(root, "linux_arm64", "3.12", REVISION)

  def test_rejects_wrong_source_revision(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      make_wheel(root, version="0.15.0+g" + "b" * 40)
      with self.assertRaisesRegex(ValueError, "source commit"):
        build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)

  def test_rejects_missing_or_ambiguous_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(ValueError, "exactly one"):
        build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)
      make_wheel(root)
      make_wheel(root, tag="manylinux_2_39_x86_64")
      with self.assertRaisesRegex(ValueError, "exactly one"):
        build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)


class DockerBuildTest(unittest.TestCase):

  def build(self):
    build_wheels.docker_build(
        ["docker", "--context", "remote"], Path("context"), Path("source.tar"),
        Path("output"), "linux_arm64", "3.13", REVISION, 4, "9.2.0")

  @mock.patch("build_wheels.subprocess.run")
  @mock.patch("build_wheels.run")
  def test_remote_build_copies_files_without_host_bind_mounts(self, run, cleanup):
    run.return_value = "0"
    self.build()
    commands = [call.args[0] for call in run.call_args_list]
    self.assertTrue(all(command[:3] == ["docker", "--context", "remote"]
                        for command in commands))
    self.assertIn("linux/arm64", commands[0])
    self.assertFalse(any("type=bind" in arg for cmd in commands for arg in cmd))
    self.assertEqual(sum(command[3] == "cp" for command in commands), 2)
    self.assertIn("--force", cleanup.call_args.args[0])

  @mock.patch("build_wheels.subprocess.run")
  @mock.patch("build_wheels.run")
  def test_failed_container_is_removed_without_exporting_artifacts(self, run, cleanup):
    run.return_value = "7"
    with self.assertRaisesRegex(RuntimeError, "status 7"):
      self.build()
    copies = [call.args[0] for call in run.call_args_list
              if call.args[0][3] == "cp"]
    self.assertEqual(len(copies), 1)
    cleanup.assert_called_once()

  @mock.patch("build_wheels.subprocess.run")
  @mock.patch("build_wheels.run")
  def test_failed_upload_still_removes_container(self, run, cleanup):
    def result(command, **_):
      if command[3] == "cp":
        raise subprocess.CalledProcessError(1, command)
    run.side_effect = result
    with self.assertRaises(subprocess.CalledProcessError):
      self.build()
    cleanup.assert_called_once()


class ArgumentsTest(unittest.TestCase):

  def test_duplicate_python_versions_are_rejected_before_building(self):
    with mock.patch("build_wheels.run") as run:
      with contextlib.redirect_stderr(io.StringIO()):
        with self.assertRaises(SystemExit):
          build_wheels.main(["--python", "3.12", "3.12"])
      run.assert_not_called()

  def test_invalid_jobs_are_rejected_before_building(self):
    with mock.patch("build_wheels.run") as run:
      with self.assertRaises(SystemExit):
        build_wheels.main(["--jobs", "0"])
      run.assert_not_called()

  def test_macos_target_requires_macos_host(self):
    with mock.patch("build_wheels.platform.system", return_value="Linux"):
      with mock.patch("build_wheels.run") as run:
        with self.assertRaises(SystemExit):
          build_wheels.main(["--platforms", "macos_arm64"])
        run.assert_not_called()

  def test_m4_defaults_include_native_macos_and_both_linux_architectures(self):
    with mock.patch("build_wheels.platform.system", return_value="Darwin"):
      with mock.patch("build_wheels.platform.machine", return_value="arm64"):
        self.assertEqual(build_wheels.defaults(),
                         ["macos_arm64", "linux_arm64", "linux_x86_64"])


class PythonVersionsTest(unittest.TestCase):

  def setUp(self):
    self.contexts = contextlib.ExitStack()
    self.addCleanup(self.contexts.close)
    self.directory = self.contexts.enter_context(tempfile.TemporaryDirectory())
    self.output = Path(self.directory)
    self.builds = []
    self.contexts.enter_context(mock.patch(
        "build_wheels.platform.system", return_value="Darwin"))
    self.contexts.enter_context(mock.patch(
        "build_wheels.platform.machine", return_value="arm64"))
    self.contexts.enter_context(mock.patch(
        "build_wheels.shutil.which", return_value="tool"))
    self.contexts.enter_context(mock.patch("build_wheels.shutil.copyfile"))
    self.contexts.enter_context(mock.patch(
        "build_wheels.run", side_effect=self.run_command))
    self.contexts.enter_context(mock.patch(
        "build_wheels.docker_build", side_effect=self.docker_build))
    self.contexts.enter_context(contextlib.redirect_stdout(io.StringIO()))
    self.contexts.enter_context(contextlib.redirect_stderr(io.StringIO()))

  def run_command(self, command, **kwargs):
    if command[:2] == ["git", "rev-parse"]:
      return REVISION
    if command[:2] == ["git", "show"]:
      if command[2].endswith(":.bazelversion"):
        return "9.2.0"
      return "WHEEL_LOCAL_VERSION"
    if command[0] == "bash":
      Path(kwargs["cwd"], "work").mkdir()
      self.write_wheel(Path(command[3]), "macos_arm64", command[4])

  def docker_build(self, docker, context, archive, output, target, python_version,
                   revision, jobs, bazel_version):
    self.write_wheel(output, target, python_version)

  def write_wheel(self, output, target, python_version):
    self.builds.append((target, python_version))
    tags = {
        "macos_arm64": "macosx_12_0_arm64",
        "linux_arm64": "manylinux_2_39_aarch64",
        "linux_x86_64": "manylinux_2_39_x86_64",
    }
    make_wheel(output, tag=tags[target], python_version=python_version)

  def launch(self, *args):
    build_wheels.main(["--output-dir", str(self.output), *args])

  def test_versions_build_on_every_platform_with_separate_manifests(self):
    self.launch("--python", "3.11", "3.12", "3.13")
    expected = {(target, version) for target in build_wheels.PLATFORMS
                for version in ("3.11", "3.12", "3.13")}
    self.assertCountEqual(self.builds, expected)
    for target, version in expected:
      directory = self.output / REVISION / target / version
      manifest = json.loads((directory / "build.json").read_text())
      self.assertEqual(manifest["python"], version)
      self.assertEqual(manifest["platform"], target)
      self.assertTrue((directory / manifest["wheel"]).is_file())

  def test_default_and_additional_version_can_share_output(self):
    self.launch("--platforms", "linux_arm64")
    self.assertEqual(self.builds, [("linux_arm64", "3.13")])
    self.launch("--platforms", "linux_arm64", "--python", "3.12")
    self.assertEqual(self.builds[-1], ("linux_arm64", "3.12"))
    with self.assertRaises(SystemExit):
      self.launch("--platforms", "linux_arm64", "--python", "3.11", "3.13")
    self.assertEqual(len(self.builds), 2)


if __name__ == "__main__":
  unittest.main()
