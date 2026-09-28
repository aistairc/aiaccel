# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

from tests.job.apps.scheduler_test_base import SchedulerTestBase


class TestPBS(SchedulerTestBase):
    cmd = ["aiaccel-job", "pbs"]
    config_path = Path(__file__).parent / "config" / "custom_pbs.yaml"
    cancel_status = 143

    def get_job_id(self, stdout: str) -> str:
        # PBS qsub:
        # 1.hostname
        job_id = stdout.strip()
        assert job_id
        return job_id

    def is_job_running(self, job_id: str) -> bool:
        result = subprocess.run(
            ["qstat", job_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def cancel_job(self, job_id: str) -> None:
        subprocess.run(
            ["qdel", job_id],
            check=True,
        )
