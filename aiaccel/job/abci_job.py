from __future__ import annotations

from typing import Any

from pathlib import Path
import re
import subprocess
import time
from xml.etree import ElementTree

from aiaccel.job.base_job import BaseJob
from aiaccel.job.job_status import JobStatus


class AbciJob(BaseJob):
    """
    Represents a job to be submitted and managed on the ABCI system.

    Attributes:
        job_filename (Path): The path to the job file.
        job_group (str): The job group.
        job_name (str): The name of the job.
        cwd (Path): The current working directory.
        stdout_filename (Path): The path to the standard output file.
        stderr_filename (Path): The path to the standard error file.
        tag (Any): A tag associated with the job.
        status (JobStatus): The status of the job.
        job_number (int | None): The job number assigned by the system.

    Methods:
        submit: Submits the job to the system.
        update_status: Updates the status of the job.
        wait: Waits for the job to finish.
        update_status_batch: Updates the status of a batch of jobs.
    """

    def __init__(
        self,
        job_filename: Path,
        job_group: str,
        job_name: str | None = None,
        cwd: Path | str | None = None,
        stdout_filename: Path | str | None = None,
        stderr_filename: Path | str | None = None,
        qsub_args: list[str] | None = None,
        args: list[str] | None = None,
        tag: Any = None,
    ):
        """
        Initializes a new instance of the AbciJob class.

        Args:
            job_filename (Path): The path to the job file.
            job_group (str): The job group.
            job_name (str | None, optional): The name of the job. If not provided, \
                the name will be derived from the job filename.
            cwd (Path | str | None, optional): The current working directory. If not provided, \
                the current working directory will be used.
            stdout_filename (Path | str | None, optional): The path to the standard output file. If not provided, \
                a default filename will be used.
            stderr_filename (Path | str | None, optional): The path to the standard error file. If not provided, \
                a default filename will be used.
            qsub_args (list[str] | None, optional): Additional arguments to pass to the qsub command. Defaults to None.
            args (list[str] | None, optional): Additional arguments to pass to the job file. Defaults to None.
            tag (Any, optional): A tag associated with the job. Defaults to None.
        """
        super().__init__(job_filename, job_name, cwd, tag)

        self.job_group = job_group

        self.stdout_filename = Path(stdout_filename) if stdout_filename is not None else self.cwd / f"{self.job_name}.o"
        self.stderr_filename = Path(stderr_filename) if stderr_filename is not None else self.cwd / f"{self.job_name}.o"

        self.job_number: int | None = None

        # generate qsub command
        self.cmd = ["qsub", "-g", job_group, "-o", str(self.stdout_filename), "-e", str(self.stderr_filename)]
        if self.job_name is not None:
            self.cmd += ["-N", self.job_name]
        if qsub_args is not None:
            self.cmd += [arg.format(job=self) for arg in qsub_args]
        self.cmd += [str(self.job_filename)]
        if args is not None:
            self.cmd += [arg.format(job=self) for arg in args]

    def submit(self) -> AbciJob:
        """
        Submits the job to the system.

        Returns:
            AbciJob: The submitted job.

        Raises:
            RuntimeError: If the job is already submitted.
            RuntimeError: If the qsub result cannot be parsed.
        """
        if self.status >= JobStatus.WAITING:
            raise RuntimeError(f"This job is already submited as {self.job_name} (id: {self.job_number})")

        p = subprocess.run(self.cmd, capture_output=True, text=True, check=True)

        match = re.search(r"Your job (\d+)", p.stdout)
        if match is None:
            raise RuntimeError(f"The following qsub result cannot be parsed: {p.stdout}")

        self.job_number = int(match.group(1))
        self.status = JobStatus.WAITING

        return self

    def wait(self, sleep_time: float = 10.0) -> AbciJob:
        """
        Waits for the job to finish.

        Args:
            sleep_time (float, optional): The time to sleep between status updates. Defaults to 10.0.

        Returns:
            AbciJob: The finished job.
        """
        while self.update_status() < JobStatus.FINISHED:
            time.sleep(sleep_time)

        return self

    @classmethod
    def update_status_batch(cls, job_list: list[AbciJob]) -> None:
        """
        Updates the status of a batch of jobs.

        Args:
            job_list (list[AbciJob]): The list of jobs to update.
        """
        job_dict = {j.job_number: j for j in job_list if j.status not in [JobStatus.UNSUBMITTED, JobStatus.FINISHED]}
        p = subprocess.run(["qstat", "-xml"], capture_output=True, text=True, check=True)

        status_dict: dict[int, str] = {}
        for el in ElementTree.fromstring(p.stdout).iter("job_list"):
            status_dict[int(el.findtext("JB_job_number", default=-1))] = el.findtext("state", "")

        for job_number in set(job_dict.keys()) - set(status_dict.keys()):
            job_dict[job_number].status = JobStatus.FINISHED

        for job_number, status in status_dict.items():
            match status:
                case "r":
                    job_dict[job_number].status = JobStatus.RUNNING
                case "qw" | "h" | "t" | "s" | "S" | "T" | "Rq":
                    job_dict[job_number].status = JobStatus.WAITING
                case "d" | "Rr":
                    job_dict[job_number].status = JobStatus.RUNNING
                case "E":
                    job_dict[job_number].status = JobStatus.ERROR
                case _:
                    raise ValueError(f"Unexpected status: {status}")
