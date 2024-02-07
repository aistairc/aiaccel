from __future__ import annotations

import sys
import time

from pathlib import Path
import subprocess

import json

from aiaccel.job.command_creator import (
    create_submit_command,
    create_execute_objective_command,
)

script_name = sys.argv[0]


class JobCreator:
    def __init__(
        self,
        trial_id: int,
        platform: str,
        group: str,
        preamble: str,
        timeout_seconds: int,
        work_dir: Path,
    ):
        self.trial_id = trial_id
        self.platform = platform
        self.group = group
        self.preamble = preamble
        self.work_dir = work_dir
        self.timeout_seconds = timeout_seconds
        self._start_time = None
        self._end_time = None
        self._returncode = None
        self.script_name = script_name
        self.job_file_path = str(self.work_dir / f"job{self.trial_id}.sh")
        self.stdout_file_path = str(self.work_dir / f"o{self.trial_id}")
        self.stderr_file_path = str(self.work_dir / f"e{self.trial_id}")

    def create(self, param: dict) -> None:
        """Create a executable file to run the job."""
        cmd = create_execute_objective_command(self.script_name, param)
        with open(self.job_file_path, "w") as f:
            f.write("#!/bin/bash\n")
            ...
            ...
            ...
            f.write(f"{cmd}\n")

    def run(self) -> None:
        """Run the job with the given hyperparameters."""
        cmd = create_submit_command(
            self.platform,
            self.script_name,
            self.group,
            self.job_file_path,
            self.stdout_file_path,
            self.stderr_file_path,
        )
        print(f"Running the job with the command: `{cmd}`")
        cmds = cmd.split()
        self._start_time = time.time()
        self._returncode = _run(
            cmds, self.stdout_file_path, self.stderr_file_path, self.timeout_seconds
        )
        self._end_time = time.time()

    def collect_result(self) -> str | None:
        """Collect the result of the job."""
        return _collect_result(self.stdout_file_path)

    def create_result_json(self, result: dict) -> None:
        """Create a json file to store the result of the job.
        The file name is `result{trial_id}.json`.
        """
        result_file_path = str(self.work_dir / f"result{self.trial_id}.json")
        with open(result_file_path, "w") as f:
            json.dump(result, f)

    def is_finished(self) -> bool:
        """Check if the job finished."""
        return self._returncode is not None

    def is_error_free(self) -> bool:
        """Check if the job finished without error."""
        return self._returncode == 0

    @property
    def returncode(self) -> int | None:
        """Get the return code of the job."""
        return self._returncode

    @property
    def start_time(self) -> float | None:
        """Get the start time of the job."""
        return self._start_time

    @property
    def end_time(self) -> float | None:
        """Get the end time of the job."""
        return self._end_time

    def get_elapsed_time(self) -> float:
        """Get the elapsed time."""
        if self._start_time is None:
            raise RuntimeError("Job not started.")
        return time.time() - self._start_time


def _run(
    cmds: list[str], stdout_file: str, stderr_file: str, timeout_seconds: float | int
) -> int:
    """Run the job with the given hyperparameters."""
    if timeout_seconds <= 0:
        result = subprocess.run(cmds, capture_output=True, text=True)
    else:
        result = subprocess.run(
            cmds, capture_output=True, text=True, timeout=timeout_seconds
        )

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
