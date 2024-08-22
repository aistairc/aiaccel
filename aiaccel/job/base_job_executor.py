from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseJobExecutor(ABC):
    """
    Abstract base class for job executors.

    Attributes:
        job_filename (Path): The path to the job file.
        job_group (str): The group to which the job belongs.
        job_name (str | None): The name of the job.
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

    @abstractmethod
    def available_slots(self) -> int:
        """
        Returns the number of available slots for new jobs.

        Returns:
            int: The number of available slots.
        """
        pass

    @abstractmethod
    def collect_finished(self) -> list[Any]:
        """
        Collects and removes all finished jobs from the job list.

        Returns:
            A list of finished job objects.
        """
        pass
