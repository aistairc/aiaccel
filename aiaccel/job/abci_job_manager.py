import time
from pathlib import Path
from typing import Any

from aiaccel.job.abci_job import AbciJob, JobStatus


class AbciJobExecutor:
    """
    Executes and manages jobs on the ABCI system.

    Attributes:
        job_filename (Path): The path to the job file.
        job_group (str): The group to which the job belongs.
        job_name (str | None): The name of the job.
        work_dir (Path): The working directory for the job.
        n_max_jobs (int): The maximum number of jobs.
        job_list (list[AbciJob]): The list of submitted jobs.

    Methods:
        submit: Submits a job to the job manager.
        available_slots: Returns the number of available slots for new jobs.
        collect_finished: Collects and removes all finished jobs from the job list.
    """

    def __init__(
        self,
        job_filename: Path | str,
        job_group: str,
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
        self.job_group = job_group
        self.job_name = job_name

        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.n_max_jobs = n_max_jobs

        self.job_list: list[AbciJob] = []

    def submit(
        self,
        args: list[str],
        tag: Any = None,
        sleep_time: float = 5.0,
    ) -> AbciJob:
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

        job = AbciJob(
            self.job_filename,
            self.job_group,
            job_name=self.job_name,
            args=args,
            tag=tag,
        )

        job.submit()
        self.job_list.append(job)

        return job

    def available_slots(self) -> int:
        """
        Returns the number of available slots for new jobs.

        The available slots are calculated by subtracting the number of finished jobs
        from the maximum number of jobs allowed.

        Returns:
            int: The number of available slots.
        """
        AbciJob.update_status_batch(self.job_list)
        return self.n_max_jobs - len([job for job in self.job_list if job.status < JobStatus.FINISHED])

    def collect_finished(self) -> list[AbciJob]:
        """
        Collects and removes all finished jobs from the job list.

        Returns:
            A list of finished AbciJob objects.
        """
        finished_jobs = [job for job in self.job_list if job.status >= JobStatus.FINISHED]
        for job in finished_jobs:
            self.job_list.remove(job)

        return finished_jobs
