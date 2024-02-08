from __future__ import annotations

from abc import ABC, abstractmethod

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
    ):
        self.script_name = script_name
        self.group = group
        self.job_file_path = job_file_path
        self.stdout_dir = stdout_dir
        self.stderr_dir = stderr_dir

    @abstractmethod
    def create_submit_command(self) -> str:
        """Create a shell command to execute the objective function."""
        raise NotImplementedError


class Local(SubmitCommandCreator):
    """For debug"""

    def create_submit_command(self) -> str:
        return f"sh {self.job_file_path}"


class Abci(SubmitCommandCreator):
    def create_submit_command(self) -> str:
        return f"qsub -g {self.group} -o {self.stdout_dir} -e {self.stderr_dir} {self.job_file_path}"


def create_submit_command(
    platform: str,
    script_name: str,
    group: str,
    job_file_path: str,
    stdout_dir: str,
    stderr_dir: str,
) -> str:
    """
    return a shell command to execute the job
    """
    if platform == __local__ or platform == "":
        return Local(
            script_name, group, job_file_path, stdout_dir, stderr_dir
        ).create_submit_command()
    elif platform == __abci__:
        return Abci(
            script_name, group, job_file_path, stdout_dir, stderr_dir
        ).create_submit_command()
    else:
        raise NotImplementedError(f"Platform '{platform}' not implemented.")


def create_execute_command(script_name: str, param: dict) -> str:
    """Create a shell command to execute the job.
    params: {
        'x': 0.5,
        'y': 0.3,
        ...
    }
    """
    args = " ".join([f"{k}={v}" for k, v in param.items()])
    cmd = f"python {script_name} -e --params {args}"
    return cmd


def create_execute_command_with_mpi4py(script_name: str, param: dict) -> str:
    """Create a shell command to execute the job with mpi4py.
    params: {
        'x': 0.5,
        'y': 0.3,
        ...
    }
    """
    args = " ".join([f"{k}={v}" for k, v in param.items()])
    cmd = f"mpiexec -n 4 python {script_name} -e --params {args}"
    return cmd
