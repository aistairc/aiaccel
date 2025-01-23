from aiaccel.hpo.results.base_result import BaseResult
from typing import Any
import pickle as pkl
from aiaccel.hpo.job_executors import BaseJobExecutor


class PickleResult(BaseResult):

    def __init__(self, filename_template: str) -> None:
        super().__init__(filename_template)

    def save(self, dst_filename: str, y: Any) -> None:
        with open(dst_filename, "wb") as f:
            pkl.dump(y, f)

    def load(self, job: BaseJobExecutor) -> int | float | str:
        with open(self.filename_template.format(job=job), "rb") as f:
            y = pkl.load(f)
        return y
