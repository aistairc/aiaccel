from typing import Protocol

from pathlib import Path


class BaseJob(Protocol):
    job_name: str
    cwd: Path


class BaseJobOutputLoader:
    def __init__(self, filename_template: str) -> None:
        self.filename_template = filename_template

    def load(self, job: BaseJob) -> int | float | str:
        raise NotImplementedError
