# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess

from tests.job.apps.base_job_tests import BaseJobTests


class BaseSchedulerTests(BaseJobTests, ABC):
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

    def assert_cpu_kill_job(self, status_path: Path) -> None:
        assert status_path is not None
        assert status_path.exists()
        assert status_path.read_text().strip() == "143"

    def assert_cpu_array(self, tmp_path: Path, log_path: Path) -> None:
        log_prefix = log_path.with_suffix("")
        for task_index in [1, 2]:
            log_file = tmp_path / f"{log_prefix.name}.{task_index}-1.log"
            assert log_file.exists()
            assert log_file.read_text().strip() == f"TASK_INDEX={task_index}"

    def assert_cpu_fails_when_log_write_fails(self, cpu_failure_stderr: str, status_path: Path) -> None:
        assert cpu_failure_stderr is not None
        assert "Job failed with 1 exit code." in cpu_failure_stderr

        assert status_path is not None
        assert status_path.exists()
        assert status_path.read_text().strip() == "1"
