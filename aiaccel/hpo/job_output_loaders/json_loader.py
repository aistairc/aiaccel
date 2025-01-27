import json

from aiaccel.hpo.job_executors import BaseJobExecutor
from aiaccel.hpo.job_output_loaders.base_loader import BaseJobOutputLoader


class JsonJobOutputLoader(BaseJobOutputLoader):
    """
    A class to handle loading results from JSON files.

    Attributes:
        filename_template (str): A template for the filename where results are stored.

    Methods:
        __init__(filename_template: str) -> None:
            Initializes the JsonJobOutputLoader with a filename template.

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
            y: int | float | str = json.load(f)
        return y
