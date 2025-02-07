from __future__ import annotations

from typing import Any, Protocol

from abc import ABC, abstractmethod
from pathlib import Path

from aiaccel.hpo.jobs.job_status import JobStatus


class JobOutputLoaderProtocol(Protocol):
    def load(self, job: BaseJob) -> int | float | str: ...


class BaseJob(ABC):
    def __init__(
        self,
        job_filename: Path,
        job_name: str | None,
        cwd: Path | str | None,
        tag: Any = None,
        loader: JobOutputLoaderProtocol | None = None,
    ):
        self.loader = loader
        self.job_filename = job_filename
        self.job_name = job_name if job_name is not None else self.job_filename.name
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.tag = tag
        self.status = JobStatus.UNSUBMITTED

    def update_status(self) -> JobStatus:
        """
        Updates the status of the job.

        Returns:
            JobStatus: The updated status of the job.
        """
        self.update_status_batch([self])
        return self.status

    @classmethod
    @abstractmethod
    def update_status_batch(cls, job_list: list[Any]) -> None:
        """
        Updates the status of a batch of jobs.

        Args:
            job_list (List[JobFuture]): The list of jobs
        """
        pass

    def load(self) -> int | float | str:
        """
        Loads the result of the job using the configured loader.

        Returns:
            int | float | str: The loaded result value.

        Raises:
            RuntimeError: If no loader is configured for this job.
        """
        if self.loader is None:
            raise RuntimeError("No loader configured for this job.")
        return self.loader.load(self)
