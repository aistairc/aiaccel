from __future__ import annotations

import subprocess
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

from aiaccel.job.base_job_executor import BaseJobExecutor
from aiaccel.job.job_status import JobStatus


def run(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
    except Exception as e:
        error_msg = f"Error executing command: {e}\n{traceback.format_exc()}"
        raise RuntimeError(error_msg) from e


class JobFuture:
    def __init__(
        self,
        future: Future[None],
        job_name: str | None = None,
        job_filename: Path | None = None,
        cwd: Path | None = None,
        tag: Any = None,
    ):
        self.future = future
        self.job_name = job_name
        self.job_filename = job_filename
        self.cwd = cwd
        self.tag = tag
        self.status = JobStatus.UNSUBMITTED

    def update_status(self) -> JobStatus:
        self.update_status_batch([self])
        return self.status

    @classmethod
    def update_status_batch(cls, job_list: list[JobFuture]) -> None:
        for job in job_list:
            if job.future.done():
                job.status = JobStatus.FINISHED
                if job.future.exception() is not None:
                    job.status = JobStatus.ERROR
            elif job.future.running():
                job.status = JobStatus.RUNNING
            else:
                job.status = JobStatus.WAITING


class LocalJobExecutor(BaseJobExecutor):
    def __init__(
        self,
        job_filename: Path | str,
        job_name: str | None = None,
        work_dir: Path | str | None = None,
        n_max_jobs: int = 5.0,
    ):
        """
        Initialize the AbciJobManager object.

        Args:
            job_filename (Path | str): The path or filename of the job.
            job_group (str): The group to which the job belongs.
            job_name (str | None, optional): The name of the job. Defaults to None.
            work_dir (Path | str | None, optional): The working directory for the job. Defaults to None.
            n_max_jobs (int, optional): The maximum number of jobs. Defaults to 100.
        """
        super().__init__(job_filename, job_name, work_dir, n_max_jobs)
        self.executor = ProcessPoolExecutor(max_workers=n_max_jobs)

        self.cwd = Path(work_dir) if work_dir is not None else Path.cwd()
        self.cwd.mkdir(parents=True, exist_ok=True)

        self.job_list: list[JobFuture] = []

    def submit(
        self,
        args: list[str],
        tag: Any = None,
        sleep_time: float = 5.0,
    ) -> JobFuture:
        """
        Submits a job to the job manager.

        Args:
            args (list[str]): The arguments for the job.
            tag (Any, optional): A tag to associate with the job. Defaults to None.
            sleep_time (float, optional): The sleep time between checking for available slots. Defaults to 5.0.

        Returns:
            AbciJob: The submitted job.
        """
        while self.available_slots() == 0:
            time.sleep(sleep_time)

        cmd = ["bash", str(self.job_filename)]
        if args is not None:
            cmd += [arg.format(job=self) for arg in args]
        print(cmd)
        future = self.executor.submit(run, cmd, self.cwd)
        job_future = JobFuture(
            future, job_name=self.job_name, job_filename=self.job_filename, cwd=self.work_dir, tag=tag
        )

        self.job_list.append(job_future)

        return job_future

    def update_status_batch(self) -> None:
        JobFuture.update_status_batch(self.job_list)
