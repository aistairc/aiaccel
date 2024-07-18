from __future__ import annotations

import subprocess
from enum import IntEnum, auto
from pathlib import Path
from typing import Any


class JobStatus(IntEnum):
    UNSUBMITTED = auto()
    RUNNING = auto()
    FINISHED = auto()
    ERROR = auto()


class LocalJob:
    job_filename: Path
    job_name: str

    cwd: Path

    tag: Any

    def __init__(
        self,
        job_filename: Path | str,
        job_name: str | None = None,
        cwd: Path | str | None = None,
        args: list[str] | None = None,
        tag: Any = None,
    ):
        self.job_filename = Path(job_filename)
        self.job_name = job_name if job_name is not None else self.job_filename.name

        self.cwd = Path(cwd) if cwd is not None else Path.cwd()

        self.tag = tag
        self.status = JobStatus.UNSUBMITTED

        self.p: subprocess.Popen[bytes] | None = None

        # generate qsub command
        self.cmd = ["bash", str(self.job_filename)]
        if args is not None:
            self.cmd += [arg.format(job=self) for arg in args]

    def submit(self) -> LocalJob:
        self.p = subprocess.Popen(self.cmd)
        self.status = JobStatus.RUNNING

        return self

    def update_status(self) -> JobStatus:
        self.update_status_batch([self])

        return self.status

    @classmethod
    def update_status_batch(cls, job_list: list[LocalJob]) -> None:
        """
        Updates the status of the job.

        Args:
            job_list (list[LocalJob]): The list of jobs to update.
        """
        for job in job_list:
            if job.status == JobStatus.UNSUBMITTED:
                continue
            if job.p is None:
                raise ValueError("Job has not been submitted.")
            if job.p.poll() is None:
                job.status = JobStatus.RUNNING
            else:
                _, _ = job.p.communicate()
                if job.p.returncode == 0:
                    job.status = JobStatus.FINISHED
                else:
                    job.status = JobStatus.ERROR
