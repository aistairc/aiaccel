# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import time

import pytest


class SchedulerTestBase(ABC):
    cmd: list[str]
    config_path: Path | None = None
    cancel_status: int

    def make_command(self) -> list[str | Path]:
        command: list[str | Path] = list(self.cmd.copy())

        if self.config_path is not None:
            command += ["--config", self.config_path]

        return command

    @abstractmethod
    def get_job_id(self, stdout: str) -> str:
        """Return the scheduler job ID from submit command output."""

    @abstractmethod
    def is_job_running(self, job_id: str) -> bool:
        """Return whether the scheduler job still exists."""

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a scheduler job."""

    def wait_for_job_to_finish(
        self,
        job_id: str,
        timeout: int = 30,
    ) -> None:
        for _ in range(timeout):
            if not self.is_job_running(job_id):
                return

            time.sleep(1)

        pytest.fail(f"Job {job_id} did not terminate")

    def test_cpu(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        log_path = tmp_path / "test.log"

        subprocess.run(
            self.make_command()
            + [
                "cpu",
                log_path,
                "--",
                "echo",
                "hello",
            ],
            check=True,
        )

        assert log_path.exists()
        assert log_path.read_text().strip().endswith("hello")

    def test_cpu_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        log_path = tmp_path / "test.log"
        status_path = tmp_path / "test.out"

        process = subprocess.Popen(
            self.make_command()
            + [
                "cpu",
                log_path,
                "--",
                "bash",
                "-c",
                "exit 7",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        assert process.stdout is not None

        job_id = self.get_job_id(process.stdout.readline())

        _, stderr = process.communicate(timeout=30)

        assert process.returncode == 1
        assert "Job failed with 7 exit code." in stderr

        self.wait_for_job_to_finish(job_id)

        assert status_path.exists()
        assert status_path.read_text().strip() == "7"

    def test_cpu_log_filename_with_spaces(
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
                log_path,
                "--",
                "echo",
                "hello",
            ],
            check=True,
        )

        assert log_path.exists()
        assert log_path.read_text().strip().endswith("hello")

    def test_cpu_array(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        log_path = tmp_path / "test_log.log"

        subprocess.run(
            self.make_command()
            + [
                "cpu",
                "--n_tasks",
                "2",
                "--n_tasks_per_proc",
                "1",
                "--n_procs",
                "1",
                log_path,
                "--",
                "bash",
                "-c",
                'echo "TASK_INDEX=$TASK_INDEX"',
            ],
            check=True,
        )

        for task_index in [1, 2]:
            log_file = tmp_path / f"test_log.{task_index}-1.log"

            assert log_file.exists()
            assert log_file.read_text().strip() == f"TASK_INDEX={task_index}"

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
                "1",
                log_path,
                "--",
                "bash",
                "-c",
                'echo "TASK_INDEX=$TASK_INDEX"',
            ],
            check=True,
        )

        for task_index in [1, 2]:
            log_file = tmp_path / f"test log.{task_index}-1.log"

            assert log_file.exists()
            assert log_file.read_text().strip() == f"TASK_INDEX={task_index}"
