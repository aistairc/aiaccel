from aiaccel.hpo.results.base_result import BaseResult
from typing import Any
from aiaccel.hpo.job_executors import BaseJobExecutor

class StdoutResult(BaseResult):

    def __init__(self, filename_template: str) -> None:
        super().__init__(filename_template)

    def load(self, job: BaseJobExecutor) -> int | float | str:
        with open(self.filename_template.format(job=job), "r") as f:
            lines = f.readlines()
            if not lines:
                raise ValueError("File is empty")

            last_line = None
            for line in reversed(lines):
                if line.strip():
                    last_line = line.strip()
                    break

            if last_line is None:
                raise ValueError("No non-empty lines found in file")

            try:
                return int(last_line)
            except ValueError:
                try:
                    return float(last_line)
                except ValueError:
                    return last_line
