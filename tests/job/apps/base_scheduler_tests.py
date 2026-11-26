# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess

from tests.job.apps.base_job_tests import BaseJobTests


class BaseSchedulerTests(BaseJobTests, ABC):
    run_cpu_array_nprocs: int = 1
    array_log_suffix = "{index}-1.log"

    @abstractmethod
    def get_job_id(self, stdout: str) -> str:
        """Return the scheduler job ID from submit command output."""

    @abstractmethod
    def is_job_running(self, job_id: str) -> bool:
        """Return whether the scheduler job still exists."""

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a scheduler job."""

    def wait_run_cpu_failure(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        job_id = self.get_job_id(process.stdout.readline())
        self.wait_for_job_to_finish(job_id)

    def assert_cpu_failure_stderr(self, cpu_failure_stderr: str, exit_code: int) -> None:
        assert cpu_failure_stderr is not None
        assert f"Job failed with {exit_code} exit code." in cpu_failure_stderr

    def assert_status_path(self, status_path: Path, exit_code: int) -> None:
        assert status_path is not None
        assert status_path.exists()
        assert status_path.read_text().strip() == f"{exit_code}"

    def assert_run_cpu_failure(self, cpu_failure_stderr: str, status_path: Path) -> None:
        self.assert_cpu_failure_stderr(cpu_failure_stderr, 7)
        self.assert_status_path(status_path, 7)

    def assert_cpu_kill_job(self, cpu_failure_stderr: str, status_path: Path) -> None:
        self.assert_cpu_failure_stderr(cpu_failure_stderr, 143)
        self.assert_status_path(status_path, 143)

    def assert_cpu_fails_when_log_write_fails(self, cpu_failure_stderr: str, status_path: Path) -> None:
        self.assert_cpu_failure_stderr(cpu_failure_stderr, 1)
        self.assert_status_path(status_path, 1)
