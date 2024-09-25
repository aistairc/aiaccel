from __future__ import annotations

import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from aiaccel.job.base_job_executor import BaseJobExecutor
from aiaccel.job.local_job import LocalJob


def run(cmd: list[str], cwd: Path) -> None:
    """
    Executes a command.

    Args:
        cmd (List[str]): The command to execute.
        cwd (Path): The current working directory.
    """
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
    except Exception as e:
        error_msg = f"Error executing command: {e}\n{traceback.format_exc()}"
        raise RuntimeError(error_msg) from e


class LocalJobExecutor(BaseJobExecutor):
    def __init__(
        self,
        job_filename: Path,
        job_name: str | None = None,
        work_dir: Path | str | None = None,
        n_max_jobs: int = 1,
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

        self.job_list: list[LocalJob] = []

    def submit(
        self,
        args: list[str],
        tag: Any = None,
        sleep_time: float = 5.0,
    ) -> LocalJob:
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

        future = self.executor.submit(run, cmd, self.cwd)
        job_future = LocalJob(
            future, job_filename=self.job_filename, job_name=self.job_name, cwd=self.work_dir, tag=tag
        )

        self.job_list.append(job_future)

        return job_future

    def update_status_batch(self) -> None:
        """
        Updates the status of a batch of jobs.
        """
        LocalJob.update_status_batch(self.job_list)
