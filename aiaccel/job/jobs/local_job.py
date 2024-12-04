from __future__ import annotations

from typing import Any

from concurrent.futures import Future
from pathlib import Path

from aiaccel.job.jobs.base_job import BaseJob
from aiaccel.job.jobs.job_status import JobStatus


class LocalJob(BaseJob):
    def __init__(
        self,
        future: Future[None],
        job_filename: Path,
        job_name: str | None = None,
        cwd: Path | None = None,
        tag: Any = None,
    ):
        super().__init__(job_filename, job_name, cwd, tag)
        self.future = future

    @classmethod
    def update_status_batch(cls, job_list: list[LocalJob]) -> None:
        """
        Updates the status of a batch of jobs.

        Args:
            job_list (List[JobFuture]): The list of jobs
        """
        for job in job_list:
            if job.future.done():
                job.status = JobStatus.FINISHED
                if job.future.exception() is not None:
                    job.status = JobStatus.ERROR
            elif job.future.running():
                job.status = JobStatus.RUNNING
            else:
                job.status = JobStatus.WAITING
