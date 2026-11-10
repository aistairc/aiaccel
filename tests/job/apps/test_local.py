# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import pytest

from tests.job.apps.job_test_base import JobTestBase


class TestLocal(JobTestBase):
    cmd = ["aiaccel-job", "local"]
    config_test_path = Path(__file__).parent / "config" / "custom_local.yaml"
    cancel_status = 143

    def run_cpu_failure_case(self, log_path: Path) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            self.make_command()
            + [
                "cpu",
                str(log_path),
                "--",
                "bash",
                "-c",
                "echo before failure; exit 7",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return process

    def assert_run_cpu_failure(self, process: subprocess.Popen[str], log_path: Path, status_path: Path) -> None:
        process.communicate(timeout=60)

        assert process.returncode == 1

        assert log_path.exists()
        assert "before failure\n" in log_path.read_text()

    def cancel_job(self, job_id: str) -> None:
        """Terminate the user program running inside the local job."""
        os.kill(int(job_id), signal.SIGTERM)

    def run_cpu_cancel_job(
        self,
        log_path: Path,
        ready_path: Path,
    ) -> subprocess.Popen[str]:
        pid_path = ready_path.with_name(f"{ready_path.name}.pid")
        command = f"echo $$ > {shlex.quote(str(pid_path))}; touch {shlex.quote(str(ready_path))}; exec sleep 10"

        process = subprocess.Popen(
            self.make_command() + ["cpu", log_path, "--", "bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        for _ in range(60):
            if ready_path.exists() and pid_path.exists():
                break

            if process.poll() is not None:
                _, stderr = process.communicate()
                pytest.fail(f"Local job exited before starting: {stderr}")

            time.sleep(1)
        else:
            process.terminate()
            process.communicate(timeout=30)
            pytest.fail("Local user program did not start")

        job_id = pid_path.read_text().strip()
        self.cancel_job(job_id)
        process.communicate(timeout=30)

        return process

    def assert_run_cpu_cancel_job(
        self, process: subprocess.Popen[str], log_path: Path, status_path: Path | None
    ) -> None:
        process.communicate(timeout=30)

        assert process.returncode == 1

        assert log_path.exists()

    def run_cpu_array(self, log_path: Path) -> None:
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

    def assert_cpu_array(self, tmp_path: Path, log_path: Path) -> None:
        log_prefix = log_path.with_suffix("")
        for process_index in [1, 2]:
            log_file = tmp_path / f"{log_prefix.name}.{process_index}.log"
            assert log_file.exists()
            assert log_file.read_text().strip() == f"TASK_INDEX={process_index}"

    def run_cpu_fails_when_log_write_fails(self, log_path: Path) -> None:
        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "echo", "job succeeded"],
            check=True,
        )

    def assert_cpu_fails_when_log_write_fails(
        self, exc_info: pytest.ExceptionInfo[subprocess.CalledProcessError], status_path: Path
    ) -> None:
        assert exc_info.value.returncode == 1
