# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest


class JobTestBase:
    cmd: list[str]
    config_path: Path | None = None
    config_test_path: Path

    def make_command(self) -> list[str | Path]:
        command: list[str | Path] = list(self.cmd.copy())

        if self.config_path is not None:
            command += ["--config", self.config_path]

        return command

    @pytest.mark.parametrize(
        "base_args",
        [
            ["cpu"],
            ["cpu", "--n_tasks=10"],
            ["gpu"],
            ["gpu", "--n_tasks=10"],
        ],
    )
    def test_default(self, base_args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"

        subprocess.run(
            self.make_command() + base_args + [log_path, "--", "sleep", "0"],
            check=True,
        )

    def test_config_from_argparse(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"

        subprocess.run(
            self.cmd + ["--config", self.config_test_path, "cpu", log_path, "sleep", "0"],
            check=True,
        )

        with open(tmp_path / "config_path.txt") as f:
            assert Path(f.read().rstrip("\n")) == self.config_test_path

    def test_config_from_environ(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"
        monkeypatch.setenv("AIACCEL_JOB_CONFIG", str(self.config_test_path))

        subprocess.run(
            self.cmd + ["cpu", log_path, "--", "sleep", "0"],
            check=True,
        )

        with open(tmp_path / "config_path.txt") as f:
            assert Path(f.read().rstrip("\n")) == self.config_test_path

    def test_cpu(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"

        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "echo", "hello"],
            check=True,
        )

        assert log_path.exists()
        assert log_path.read_text().strip().endswith("hello")

    # TODO: Add tests for log filenames containing shell metacharacters
    # and other characters that require escaping.
    def test_cpu_log_filename_with_spaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test log.log"

        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "echo", "hello"],
            check=True,
        )

        assert log_path.exists()
        assert log_path.read_text().strip().endswith("hello")

    def test_cpu_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"
        status_path = tmp_path / "test.out"

        process = self.run_cpu_failure_case(log_path)

        self.assert_run_cpu_failure(process, log_path, status_path)

    def run_cpu_failure_case(self, log_path: Path) -> subprocess.Popen[str]:
        raise NotImplementedError

    def assert_run_cpu_failure(self, process: subprocess.Popen[str], log_path: Path, status_path: Path) -> None:
        raise NotImplementedError

    def test_cpu_cancel_job(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        log_path = tmp_path / "test.log"
        status_path = tmp_path / "test.out"
        ready_path = tmp_path / "ready"

        process = self.run_cpu_cancel_job(log_path, ready_path)

        self.assert_run_cpu_cancel_job(process, log_path, status_path)

    def run_cpu_cancel_job(self, log_path: Path, ready_path: Path) -> subprocess.Popen[str]:
        raise NotImplementedError

    def assert_run_cpu_cancel_job(
        self, process: subprocess.Popen[str], log_path: Path, status_path: Path | None
    ) -> None:
        raise NotImplementedError

    def test_cpu_array(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test_log.log"

        self.run_cpu_array(log_path)

        self.assert_cpu_array(tmp_path)

    def run_cpu_array(self, log_path: Path) -> None:
        raise NotImplementedError

    def assert_cpu_array(self, tmp_path: Path) -> None:
        raise NotImplementedError

    def test_cpu_array_log_filename_with_spaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test log.log"

        subprocess.run(
            self.make_command()
            + [
                "cpu",
                "--n_tasks",
                "2",
                "--n_tasks_per_proc",
                "1",
                "--n_procs",
                "2",
                log_path,
                "--",
                "bash",
                "-c",
                'echo "TASK_INDEX=$TASK_INDEX"',
            ],
            check=True,
        )

        self.assert_cpu_array_log_filename_with_spaces(tmp_path)

    def assert_cpu_array_log_filename_with_spaces(self, tmp_path: Path) -> None:
        raise NotImplementedError
