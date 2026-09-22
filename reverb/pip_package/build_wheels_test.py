"""Check artifact validation and Docker failure handling without a compiler."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import build_wheels


REVISION = "a" * 40


class WheelValidationTest(unittest.TestCase):

  def wheel(self, directory, *, version=None, tag="manylinux_2_39_aarch64"):
    version = version or "0.15.0+g" + REVISION
    path = directory / f"dm_reverb-{version}-cp313-cp313-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
      archive.writestr(
          f"dm_reverb-{version}.dist-info/METADATA",
          f"Name: dm-reverb\nVersion: {version}\n")
    return path

  def test_matching_wheel(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      wheel = self.wheel(root)
      self.assertEqual(
          build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)[0],
          wheel)

  def test_rejects_wrong_architecture(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.wheel(root, tag="manylinux_2_39_x86_64")
      with self.assertRaisesRegex(ValueError, "tags"):
        build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)

  def test_rejects_wrong_python(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.wheel(root)
      with self.assertRaisesRegex(ValueError, "tags"):
        build_wheels.check_wheel(root, "linux_arm64", "3.12", REVISION)

  def test_rejects_wrong_source_revision(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.wheel(root, version="0.15.0+g" + "b" * 40)
      with self.assertRaisesRegex(ValueError, "source commit"):
        build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)

  def test_rejects_missing_or_ambiguous_artifacts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(ValueError, "exactly one"):
        build_wheels.check_wheel(root, "linux_arm64", "3.13", REVISION)
      self.wheel(root)
      self.wheel(root, tag="manylinux_2_39_x86_64")
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


if __name__ == "__main__":
  unittest.main()
