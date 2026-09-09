# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import subprocess

from tests.job.apps.scheduler_test_base import SchedulerTestBase


class TestSGE(SchedulerTestBase):
    cmd = ["aiaccel-job", "sge"]
    config_path = Path(__file__).parent / "config" / "custom_sge.yaml"
    cancel_status = 140

    def get_job_id(self, stdout: str) -> str:
        # SGE qsub:
        # Your job 3 ("test") has been submitted
        match = re.search(r"Your job (\d+)", stdout)
        assert match is not None
        return match.group(1)

    def is_job_running(self, job_id: str) -> bool:
        result = subprocess.run(
            ["qstat", "-j", job_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def cancel_job(self, job_id: str) -> None:
        subprocess.run(
            ["qdel", job_id],
            check=True,
        )
