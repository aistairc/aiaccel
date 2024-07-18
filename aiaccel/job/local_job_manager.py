import time
from pathlib import Path
from typing import Any

from aiaccel.job.local_job import JobStatus, LocalJob


class LocalJobExecutor:
    def __init__(
        self,
        job_filename: Path | str,
        job_name: str | None = None,
        work_dir: Path | str | None = None,
        n_max_jobs: int = 100,
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

        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.n_max_jobs = n_max_jobs

        self.job_list: list[LocalJob] = []

    def available_slots(self) -> int:
        LocalJob.update_status_batch(self.job_list)
        return self.n_max_jobs - len([job for job in self.job_list if job.status < JobStatus.FINISHED])

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

        job = LocalJob(
            self.job_filename,
            job_name=self.job_name,
            args=args,
            tag=tag,
        )

        job.submit()
        self.job_list.append(job)

        return job

    def collect_finished(self) -> list[LocalJob]:
        """
        Collects and removes all finished jobs from the job list.

        Returns:
            A list of finished AbciJob objects.
        """
        finished_jobs = [job for job in self.job_list if job.status == JobStatus.FINISHED]
        for job in finished_jobs:
            self.job_list.remove(job)

        return finished_jobs
