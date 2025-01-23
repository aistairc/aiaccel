from typing import Any

import pickle as pkl

from aiaccel.hpo.job_executors import BaseJobExecutor
from aiaccel.hpo.results.base_result import BaseResult


class PickleResult(BaseResult):

    def __init__(self, filename_template: str) -> None:
        super().__init__(filename_template)

    def load(self, job: BaseJobExecutor) -> int | float | str:
        with open(self.filename_template.format(job=job), "rb") as f:
            y: int | float | str = pkl.load(f)
        return y
