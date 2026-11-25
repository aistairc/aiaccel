# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import time

import pytest

from tests.job.apps.base_job_tests import BaseJobTests


class BaseSchedulerTests(BaseJobTests, ABC):
    cancel_status: int
    run_cpu_array_nprocs: int = 1

    @abstractmethod
    def get_job_id(self, stdout: str) -> str:
        """Return the scheduler job ID from submit command output."""

    @abstractmethod
    def is_job_running(self, job_id: str) -> bool:
        """Return whether the scheduler job still exists."""

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a scheduler job."""

    def wait_for_job_to_finish(self, job_id: str, timeout: int = 30) -> None:
        for _ in range(timeout):
            if not self.is_job_running(job_id):
                return

            time.sleep(1)

        pytest.fail(f"Job {job_id} did not terminate")

    def wait_run_cpu_failure(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        job_id = self.get_job_id(process.stdout.readline())
        self.wait_for_job_to_finish(job_id)

    def assert_run_cpu_failure(self, cpu_failure_stderr: str, status_path: Path) -> None:
        assert cpu_failure_stderr is not None
        assert "Job failed with 7 exit code." in cpu_failure_stderr

        assert status_path is not None
        assert status_path.exists()
        assert status_path.read_text().strip() == "7"

    def run_cpu_cancel_job(self, log_path: Path, ready_path: Path) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            self.make_command()
            + [
                "cpu",
                log_path,
                "--",
                "bash",
                "-c",
                f"touch {ready_path}; sleep 10; exit 0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        assert process.stdout is not None

        job_id = self.get_job_id(process.stdout.readline())

        for _ in range(60):
            if ready_path.exists():
                break

            time.sleep(1)
        else:
            self.cancel_job(job_id)
            pytest.fail(f"Job {job_id} did not start")

        self.cancel_job(job_id)
        self.wait_for_job_to_finish(job_id)

        return process

    def assert_run_cpu_cancel_job(
        self, process: subprocess.Popen[str], log_path: Path, status_path: Path | None
    ) -> None:
        _, stderr = process.communicate(timeout=30)

        assert process.returncode == 1
        assert f"Job failed with {self.cancel_status} exit code." in stderr

        assert log_path.exists()

        assert status_path is not None
        assert status_path.exists()
        assert status_path.read_text().strip() == str(self.cancel_status)

    def assert_cpu_array(self, tmp_path: Path, log_path: Path) -> None:
        log_prefix = log_path.with_suffix("")
        for task_index in [1, 2]:
            log_file = tmp_path / f"{log_prefix.name}.{task_index}-1.log"
            assert log_file.exists()
            assert log_file.read_text().strip() == f"TASK_INDEX={task_index}"

    def assert_cpu_fails_when_log_write_fails(
        self, exc_info: pytest.ExceptionInfo[subprocess.CalledProcessError], status_path: Path
    ) -> None:
        assert "Job failed with 1 exit code." in exc_info.value.stderr
        assert status_path.exists()
        assert status_path.read_text().strip() == "1"
