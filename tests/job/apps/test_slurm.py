# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess
import time

import pytest

from tests.job.apps.scheduler_test_base import SchedulerTestBase


class TestSlurm(SchedulerTestBase):
    cmd = ["aiaccel-job", "slurm"]
    config_path = Path(__file__).parent / "config" / "custom_slurm.yaml"
    cancel_status = 143

    def get_job_id(self, stdout: str) -> str:
        # Submitted batch job 3
        fields = stdout.split()
        assert fields
        return fields[-1]

    def is_job_running(self, job_id: str) -> bool:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())

    def cancel_job(self, job_id: str) -> None:
        subprocess.run(
            ["scancel", job_id],
            check=True,
        )

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

        for _ in range(40):
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T %R"],
                capture_output=True,
                text=True,
                check=True,
            )

            print(f"{job_id=}, {result.stdout=!r}")

            if result.stdout.startswith("RUNNING"):
                break

            time.sleep(1)

        _, stderr = process.communicate(timeout=30)

        assert process.returncode == 1
        assert "Job failed with 7 exit code." in stderr

        self.wait_for_job_to_finish(job_id)

        assert status_path.exists()
        assert status_path.read_text().strip() == "7"
