# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

from tests.job.apps.scheduler_test_base import SchedulerTestBase


class TestSlurm(SchedulerTestBase):
    cmd = ["aiaccel-job", "slurm"]
    config_path = Path(__file__).parent / "config" / "custom_slurm.yaml"
    cancel_status = 143

    def get_job_id(self, stdout: str) -> str:
        # Slurm sbatch:
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
