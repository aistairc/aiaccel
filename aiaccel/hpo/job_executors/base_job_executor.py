from typing import Any

from abc import ABC, abstractmethod
from pathlib import Path

from aiaccel.hpo.jobs.abci_job import AbciJob
from aiaccel.hpo.jobs.base_job import BaseJob
from aiaccel.hpo.jobs.job_status import JobStatus
from aiaccel.hpo.jobs.local_job import LocalJob


class BaseJobExecutor(ABC):
    JobClass: type[BaseJob | AbciJob | LocalJob]

    """
    Abstract base class for job executors.

    Attributes:
        job_filename (Path): The path to the job file.
        work_dir (Path): The working directory for the job.
        n_max_jobs (int): The maximum number of jobs.
        job_list (List[Any]): The list of submitted jobs.

    Methods:
        submit: Submits a job to the job manager.
        available_slots: Returns the number of available slots for new jobs.
        collect_finished: Collects and removes all finished jobs from the job list.
    """

    def __init__(
        self,
        job_filename: Path | str,
        job_name: str | None = None,
        work_dir: Path | str | None = None,
        n_max_jobs: int = 100,
    ):
        self.job_filename = job_filename if isinstance(job_filename, Path) else Path(job_filename)
        self.job_name = job_name
        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.n_max_jobs = n_max_jobs

        self.job_list: list[Any] = []

    @abstractmethod
    def submit(
        self,
        args: list[str],
        tag: Any = None,
        sleep_time: float = 5.0,
    ) -> Any:
        """
        Submits a job to the job manager.

        Args:
            args (List[str]): The arguments for the job.
            tag (Any, optional): A tag to associate with the job. Defaults to None.
            sleep_time (float, optional): The sleep time between checking for available slots. Defaults to 5.0.

        Returns:
            Any: The submitted job.
        """
        pass

    def update_status_batch(self) -> None:
        """
        Updates the status of a batch of jobs.
        """
        self.JobClass.update_status_batch(self.job_list)

    def available_slots(self) -> int:
        """
        Returns the number of available slots for new jobs.

        Returns:
            int: The number of available slots.
        """
        self.update_status_batch()
        return self.n_max_jobs - len([job for job in self.job_list if job.status < JobStatus.FINISHED])

    def collect_finished(self) -> list[Any]:
        """
        Collects and removes all finished jobs from the job list.

        Returns:
            A list of finished Job objects.
        """
        finished_jobs = [job for job in self.job_list if job.status >= JobStatus.FINISHED]
        for job in finished_jobs:
            self.job_list.remove(job)

        return finished_jobs

    def get_running_jobs(self) -> list[Any]:
        """
        Returns a list of running jobs.

        Returns:
            list[JobFuture]: A list of running jobs.
        """
        return [job for job in self.job_list if job.status == JobStatus.RUNNING]
