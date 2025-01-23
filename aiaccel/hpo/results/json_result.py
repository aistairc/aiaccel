from aiaccel.hpo.results.base_result import BaseResult
from typing import Any
import json
from aiaccel.hpo.job_executors import BaseJobExecutor


class JsonResult(BaseResult):

    def __init__(self, filename_template: str) -> None:
        super().__init__(filename_template)

    def load(self, job: BaseJobExecutor) -> int | float | str:
        with open(self.filename_template.format(job=job), "r") as f:
            y = json.load(f)
        return y["objective"]
