from __future__ import annotations

import subprocess

from typing import Any
from optuna.trial import Trial


class Local:
    def __init__(self, objective: str, job_file: str, stdout_file: str, stderr_file: str):
        self.objective = objective
        self.job_file = job_file
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file

    def generate_objective_command(self) -> str:
        """Create a shell command to execute the job.
        """
        cmd = f"python {self.objective} --e"  # TODO: add hyperparameters
        return cmd

    def generate_submit_command(self) -> list[str]:
        """Create a shell command to execute the objective function.
        """
        cmd = ["sh", f"{self.job_file}"]
        return cmd

    def create(self, trial: Trial) -> None:
        """Create a hyperparameter object.
        """
        cmd = self.generate_objective_command()
        with open(self.job_file, "w") as f:
            f.write("#!/bin/bash\n")
            ...
            ...
            ...
            f.write(f"{cmd}\n")

    def run(self) -> None:
        """Run the job with the given hyperparameters.
        """
        cmds = self.generate_submit_command()
        _run(cmds, self.stdout_file, self.stderr_file)

    def collect_result(self) -> Any | None:
        """Collect the result of the job.

        return:
            The result of the job (objective value).
        """
        return _collect_result(self.stdout_file)

    ...


class Abci(Local):
    def __init__(self, objective: str, group: str, job_file: str, stdout_file: str, stderr_file: str):
        self.objective = objective
        self.group = group
        self.job_file = job_file
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file

    def generate_submit_command(self) -> str:
        """Create a shell command to execute the job.
        """
        cmd = f"qsub -g {self.group} -o {self.stdout_file} -e {self.stderr_file} {self.job_file}"
        return cmd

    ...


def _run(cmds: list[str], stdout_file: str, stderr_file: str) -> None:
    """Run the job with the given hyperparameters.
    """
    print(f"Running the job with the command: {cmds}")
    result = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    with open(stdout_file, "w") as f:
        f.write(result.stdout)

    with open(stderr_file, "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Failed to submit the job.")


def _collect_result(stdout_file: str) -> None | Any:
    """Collect the result of the job.

    return:
        The result of the job (objective value).
    """
    try:
        with open(stdout_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if lines:
                return lines[-1].strip()
            else:
                return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
