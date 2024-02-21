from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Callable

__local__ = "local"  # for debug
__abci__ = "abci"


class SubmitCommandCreator(ABC):
    def __init__(
        self,
        script_name: str,
        group: str,
        job_file_path: str,
        stdout_dir: str,
        stderr_dir: str,
        hparams_str: str,
    ):
        self.script_name = script_name
        self.group = group
        self.job_file_path = job_file_path
        self.stdout_dir = stdout_dir
        self.stderr_dir = stderr_dir
        self.hparams_str = hparams_str

    @abstractmethod
    def create_submit_command(self) -> str:
        """Create a shell command to execute the objective function."""
        raise NotImplementedError


class Local(SubmitCommandCreator):
    """For debug"""

    def create_submit_command(self) -> str:
        return f"bash {self.job_file_path} {self.hparams_str}"


class Abci(SubmitCommandCreator):
    def create_submit_command(self) -> str:
        return f"qsub -g {self.group} -o {self.stdout_dir} -e {self.stderr_dir} {self.job_file_path} {self.hparams_str}"


def create_submit_command(
    platform: str,
    script_name: str,
    group: str,
    job_file_path: str,
    stdout_dir: str,
    stderr_dir: str,
    hparams_str: str,
) -> str:
    """
    return a shell command to execute the job
    """
    if platform == __local__ or platform == "":
        return Local(
            script_name, group, job_file_path, stdout_dir, stderr_dir, hparams_str
        ).create_submit_command()
    elif platform == __abci__:
        return Abci(
            script_name, group, job_file_path, stdout_dir, stderr_dir, hparams_str
        ).create_submit_command()
    else:
        raise NotImplementedError(f"Platform '{platform}' not implemented.")


def create_execute_command(
    execute_cmd: str | None, script_name: str, python_execute_cmd: str
) -> str:
    """Create a shell command to execute the job."""
    if execute_cmd is None:
        cmd = f"{python_execute_cmd} {script_name} -e --params $@"
    else:
        cmd = f"{execute_cmd} $@"
    return cmd


def create_execute_command_with_mpi4py(
    execute_cmd: str | None, script_name: str, n_procs: int, python_execute_cmd: str
) -> str:
    """Create a shell command to execute the job with mpi4py."""
    if execute_cmd is None:
        cmd = f"mpiexec -n {n_procs} {python_execute_cmd} {script_name} -e --params $@"
    else:
        cmd = f"mpiexec -n {n_procs} {execute_cmd} $@"
    return cmd
