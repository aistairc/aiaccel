from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from aiaccel.job.abci_job import AbciJob, JobStatus


class AbciJobExecutor:
    def __init__(
        self,
        job_filename: Path | str,
        job_group: str,
        job_name: str | None = None,
        work_dir: Path | str | None = None,
        n_max_jobs: int = 100,
    ):
        self.job_filename = job_filename if isinstance(job_filename, Path) else Path(job_filename)
        self.job_group = job_group
        self.job_name = job_name

        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.n_max_jobs = n_max_jobs

        self.job_list: list[AbciJob] = []
        self.finished_job_list: list[AbciJob] = []

    def submit(
        self,
        args: list[str],
        tag: Any = None,
        sleep_time: float = 5.0,
    ) -> AbciJob:
        job = AbciJob(
            self.job_filename,
            self.job_group,
            job_name=self.job_name,
            args=args,
            tag=tag,
        )

        job.submit()
        self.job_list.append(job)

        while self.available_slots() == 0:
            time.sleep(sleep_time)

        return job

    def available_slots(self) -> int:
        AbciJob.update_status_batch(self.job_list)
        return self.n_max_jobs - len([job for job in self.job_list if job.status < JobStatus.FINISHED])

    def collect_finished(self) -> list[AbciJob]:
        finished_jobs = [job for job in self.job_list if job.status >= JobStatus.FINISHED]
        for job in finished_jobs:
            self.job_list.remove(job)
            self.finished_job_list.append(job)

        return finished_jobs

    def get_finished_job_count(self) -> int:
        return len(self.finished_job_list)
