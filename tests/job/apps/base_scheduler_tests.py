# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import time

import pytest

from tests.job.apps.base_job_tests import BaseJobTests


class BaseSchedulerTests(BaseJobTests, ABC):
    array_n_procs: int = 1
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

    def wait_for_job_to_finish(
        self, process: subprocess.Popen[str], job_id: str | None = None, timeout: int = 30
    ) -> tuple[str, str]:
        assert process.stdout is not None
        if job_id is None:
            job_id = self.get_job_id(process.stdout.readline())

        output = super().wait_for_job_to_finish(process)

        for _ in range(timeout):
            if not self.is_job_running(job_id):
                return output

            time.sleep(1)

        pytest.fail(f"Job {job_id} did not terminate")

    def assert_cpu_failure(self, cpu_failure_stderr: str, status_path: Path, exit_code: int) -> None:
        assert cpu_failure_stderr is not None
        assert f"Job failed with {exit_code} exit code." in cpu_failure_stderr

        assert status_path is not None
        assert status_path.exists()
        assert status_path.read_text().strip() == f"{exit_code}"

    def assert_run_cpu_failure(self, cpu_failure_stderr: str, status_path: Path) -> None:
        self.assert_cpu_failure(cpu_failure_stderr, status_path, 7)

    def assert_cpu_kill_job(self, cpu_failure_stderr: str, status_path: Path) -> None:
        self.assert_cpu_failure(cpu_failure_stderr, status_path, 143)

    def assert_cpu_fails_when_log_write_fails(self, cpu_failure_stderr: str, status_path: Path) -> None:
        self.assert_cpu_failure(cpu_failure_stderr, status_path, 1)
