
import json

from aiaccel.hpo.job_executors import BaseJobExecutor
from aiaccel.hpo.results.base_result import BaseResult


class JsonResult(BaseResult):
    """
    A class to handle loading results from JSON files.

    Attributes:
        filename_template (str): A template for the filename where results are stored.

    Methods:
        __init__(filename_template: str) -> None:
            Initializes the JsonResult with a filename template.

        load(job: BaseJobExecutor) -> int | float | str:
            Loads the result from a JSON file based on the job information.
    """

    def __init__(self, filename_template: str) -> None:
        super().__init__(filename_template)

    def load(self, job: BaseJobExecutor) -> int | float | str:
        """
        Loads the objective value from a JSON file for the given job.

        Args:
            job (BaseJobExecutor): The job executor instance containing job-specific information.

        Returns:
            int | float | str: The objective value extracted from the JSON file.
        """
        with open(self.filename_template.format(job=job)) as f:
            data = json.load(f)
            y: int | float | str = data["objective"]
        return y
