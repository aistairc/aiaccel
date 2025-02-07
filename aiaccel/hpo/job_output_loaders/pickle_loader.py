import pickle as pkl

from aiaccel.hpo.job_output_loaders.base_loader import BaseJob, BaseJobOutputLoader


class PickleJobOutputLoader(BaseJobOutputLoader):
    """
    A class to handle the loading of results from pickle files.

    Attributes:
        filename_template (str): A template for the filename where results are stored.

    Methods:
        __init__(filename_template: str) -> None:
            Initializes the PickleJobOutputLoader with a filename template.

        load(job: BaseJob) -> int | float | str:
            Loads the result from a pickle file corresponding to the given job.
    """

    def __init__(self, filename_template: str) -> None:
        super().__init__(filename_template)

    def load(self, job: BaseJob) -> int | float | str:
        """
        Loads the result of a job from a pickle file.

        Args:
            job (BaseJobExecutor): The job executor instance for which the result is to be loaded.

        Returns:
            int | float | str: The result of the job, which can be an integer, float, or string.
        """
        with open(self.filename_template.format(job=job), "rb") as f:
            y: int | float | str = pkl.load(f)
        return y
