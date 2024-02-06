from __future__ import annotations

import subprocess


class Local:
    def __init__(
        self,
        script_name: str,
        preamble: str,
        job_file_path: str,
        stdout_file_path: str,
        stderr_file_path: str,
    ):
        self.script_name = script_name
        self.preamble = preamble
        self.job_file_path = job_file_path
        self.stdout_file_path = stdout_file_path
        self.stderr_file_path = stderr_file_path

    def generate_objective_command(self, param: dict) -> str:
        """Create a shell command to execute the job.
        params: {
            'x': 0.5,
            'y': 0.3,
            ...
        }
        """
        args = " ".join([f"{k}={v}" for k, v in param.items()])
        cmd = f"python {self.script_name} -e --params {args}"
        return cmd

    def generate_submit_command(self) -> list[str]:
        """Create a shell command to execute the objective function."""
        cmd = ["sh", f"{self.job_file_path}"]
        return cmd

    def create(self, param: dict) -> None:
        """Create a executable file to run the job."""
        cmd = self.generate_objective_command(param)
        with open(self.job_file_path, "w") as f:
            f.write("#!/bin/bash\n")
            ...
            ...
            ...
            f.write(f"{cmd}\n")

    def run(self) -> int:
        """Run the job with the given hyperparameters.

        return:
            int: The result of the job (objective value).
        """
        cmds = self.generate_submit_command()
        print(f"Running the job with the command: `{' '.join(cmds)}`")
        return _run(cmds, self.stdout_file_path, self.stderr_file_path)

    def collect_result(self) -> str | None:
        """Collect the result of the job.

        return:
            The result of the job (objective value).
        """
        return _collect_result(self.stdout_file_path)

    ...


class Abci(Local):
    def __init__(
        self,
        script_name: str,
        preamble: str,
        group: str,
        job_file_path: str,
        stdout_dir: str,
        stderr_dir: str,
    ):
        self.script_name = script_name
        self.preamble = preamble
        self.group = group
        self.job_file_path = job_file_path
        self.stdout_dir = stdout_dir
        self.stderr_dir = stderr_dir

    def generate_submit_command(self) -> str:
        """Create a shell command to execute the job."""
        cmd = f"qsub -g {self.group} -o {self.stdout_dir} -e {self.stderr_dir} {self.job_file}"
        return cmd

    ...


def _run(cmds: list[str], stdout_file: str, stderr_file: str) -> int:
    """Run the job with the given hyperparameters."""
    result = subprocess.run(cmds, capture_output=True, text=True)

    with open(stdout_file, "w") as f:
        f.write(result.stdout)

    with open(stderr_file, "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Failed to submit the job.")

    return result.returncode


def _collect_result(stdout_file: str) -> str | None:
    """Collect the result of the job.

    return:
        The result of the job (objective value).
    """
    try:
        with open(stdout_file, encoding="utf-8") as file:
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
