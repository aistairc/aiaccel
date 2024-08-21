import subprocess
import sys
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any


def run(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
    except Exception as e:
        error_msg = f"Error executing command: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr, flush=True)
        raise


class LocalJobExecutor:
    def __init__(
        self,
        job_filename: Path | str,
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
        self.job_filename = job_filename if isinstance(job_filename, Path) else Path(job_filename)
        self.job_name = job_name if job_name is not None else self.job_filename.name

        self.cwd = Path(work_dir) if work_dir is not None else Path.cwd()
        self.cwd.mkdir(parents=True, exist_ok=True)

        self.n_max_jobs = n_max_jobs
        self.executor = ProcessPoolExecutor(max_workers=n_max_jobs)

        self.job_list: list[Future[None]] = []

    def available_slots(self) -> int:
        """
        Returns the number of available slots for new jobs.

        Returns:
            int: The number of available job slots.
        """
        return self.n_max_jobs - len(self.job_list)

    def submit(
        self,
        args: list[str],
        tag: Any = None,
        sleep_time: float = 5.0,
    ) -> Future[None]:
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

        future.job_name = self.job_name  # type: ignore
        future.job_filename = self.job_filename  # type: ignore
        future.cwd = self.cwd  # type: ignore
        future.tag = tag  # type: ignore

        self.job_list.append(future)

        return future

    def collect_finished(self) -> list[Future[None]]:
        """
        Collects and removes all finished jobs from the job list.

        Returns:
            A list of finished jobs.
        """
        finished_jobs = []
        still_running = []

        for future in list(self.job_list):
            if future.done():
                future.result()
                finished_jobs.append(future)
            else:
                still_running.append(future)

        self.job_list = still_running

        return finished_jobs
