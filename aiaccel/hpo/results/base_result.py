from aiaccel.hpo.job_executors import BaseJobExecutor


class BaseResult:
    def __init__(self, filename_template: str) -> None:
        self.filename_template = filename_template

    def load(self, job: BaseJobExecutor) -> int | float | str:
        raise NotImplementedError
