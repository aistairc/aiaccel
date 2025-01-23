from aiaccel.hpo.job_executors import BaseJobExecutor
from aiaccel.hpo.results.base_result import BaseResult


class StdoutResult(BaseResult):
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
