import ast

from aiaccel.hpo.job_executors import BaseJobExecutor
from aiaccel.hpo.job_output_loaders.base_loader import BaseJobOutputLoader


class StdoutJobOutputLoader(BaseJobOutputLoader):
    """
    A class to handle the loading of results from a stdout file.

    Args:
        filename_template (str): A template for the filename where the results are stored.

    Methods:
        load(job: BaseJobExecutor) -> int | float | str:
            Loads the result from the file corresponding to the given job.
            The result is expected to be on the last non-empty line of the file.
            Returns the result as an integer, float, or string, depending on the content.
            Raises a ValueError if the file is empty or contains no non-empty lines.
    """

    def __init__(self, filename_template: str) -> None:
        super().__init__(filename_template)

    def load(self, job: BaseJobExecutor) -> int | float | str:
        """
        Loads the result from a file associated with the given job.

        This method reads the file specified by `self.filename_template` formatted with the given job.
        It reads the file line by line in reverse order to find the last non-empty line.
        The method attempts to convert the last non-empty line to an integer, then to a float,
        and if both conversions fail, it returns the line as a string.

        Args:
            job (BaseJobExecutor): The job executor instance containing job-specific information.

        Returns:
            int | float | str: The value from the last non-empty line of the file, converted to an int,
                               float, or returned as a string.

        Raises:
            ValueError: If the file is empty or contains no non-empty lines.
        """
        with open(self.filename_template.format(job=job)) as f:
            text = f.read()

            if text is None:
                raise ValueError("No non-empty lines found in file")

            try:
                data: int | float | str = ast.literal_eval(text)
                return data
            except (ValueError, SyntaxError):
                return text
