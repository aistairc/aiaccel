from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Callable

__local__ = "local"  # for debug
__abci__ = "abci"


class SubmitCommandCreator(ABC):
    def __init__(
        self,
        base_job_file_path: str,
        group: str,
        job_file_path: str,
        stdout_path: str,
        stderr_path: str,
        hparams_str: str,
    ):
        self.base_job_file_path = base_job_file_path
        self.group = group
        self.job_file_path = job_file_path
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.hparams_str = hparams_str

    @abstractmethod
    def create_submit_command(self) -> str:
        """Create a shell command to execute the objective function."""
        raise NotImplementedError


class Local(SubmitCommandCreator):
    """For debug"""

    def create_submit_command(self) -> str:
        return f"bash {self.base_job_file_path} {self.hparams_str}"


class Abci(SubmitCommandCreator):
    def create_submit_command(self) -> str:
        if self.group == "":
            raise ValueError("Group name is required for ABCI.")
        # return f"qsub -g {self.group} -o {self.stdout_path} -e {self.stderr_path} {self.base_job_file_path} {self.hparams_str}"
        return f"qsub -g {self.group} -o {self.stdout_path} -e {self.stderr_path} {self.job_file_path} {self.hparams_str}"


def create_submit_command(
    platform: str,
    base_job_file_path: str,
    group: str,
    job_file_path: str,
    stdout_path: str,
    stderr_path: str,
    hparams_str: str,
) -> str:
    """
    return a shell command to execute the job
    """
    if platform == __local__ or platform == "":
        return Local(
            base_job_file_path,
            group,
            job_file_path,
            stdout_path,
            stderr_path,
            hparams_str,
        ).create_submit_command()
    elif platform == __abci__:
        return Abci(
            base_job_file_path,
            group,
            job_file_path,
            stdout_path,
            stderr_path,
            hparams_str,
        ).create_submit_command()
    else:
        raise NotImplementedError(f"Platform '{platform}' not implemented.")
