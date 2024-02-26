from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from aiaccel.job.command_creator import create_execute_command, create_submit_command

from typing import Callable
import fcntl
from functools import wraps
from typing import Any


def retry(_MAX_NUM: int = 60, _DELAY: float = 1.0) -> Callable[[Any], Any]:
    """Decorator to retry function.

    Args:
        _MAX_NUM (int, optional): Maximum number of retries. Defaults to 60.
        _DELAY (float, optional): Retry interval in seconds. Defaults to 1.0.
    """

    def _retry(func: Callable[[Any], Any]) -> Any:
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            for i in range(_MAX_NUM):
                try:
                    return func(*args, **kwargs)
                except BaseException as e:
                    if i == _MAX_NUM - 1:
                        raise e
                    time.sleep(_DELAY)
                    continue

        return _wrapper

    return _retry


class JobCreator:
    def __init__(
        self,
        script_name: str,
        job_name: int,
        execute_cmd: str | None,
        python_execute_cmd: str,
        platform: str,
        group: str,
        template: str,
        template_file: Path | None,
        timeout_seconds: int,
        work_dir: Path,
    ):
        self.script_name = script_name
        self.job_name = job_name
        self.execute_cmd = execute_cmd
        self.python_execute_cmd = python_execute_cmd
        self.platform = platform
        self.group = group
        self.template = template
        self.template_file = template_file
        self.work_dir = work_dir
        self.timeout_seconds = timeout_seconds
        self._start_time = None
        self._end_time = None
        self._returncode = None
        self.job_file_path = str(self.work_dir / f"{self.job_name}.sh")
        self.stdout_file_path = str(self.work_dir / f"{self.job_name}.o")
        self.stderr_file_path = str(self.work_dir / f"{self.job_name}.e")
        self.lock_file_path = str(self.work_dir / f"{self.job_name}.lock")

    def create(self) -> None:
        """Create a executable file to run the job."""
        cmd = create_execute_command(
            self.execute_cmd, self.script_name, self.python_execute_cmd
        )
        template = ""
        if self.template_file is not None:
            if self.template_file.exists():
                with open(self.template_file, "r") as f:
                    template = f.read()
            else:
                raise FileNotFoundError(
                    f"template file not found: {self.template_file}"
                )
        else:
            template = self.template

        with open(self.job_file_path, "w") as f:
            if template:
                f.write(f"{template}\n")
            f.write(f"LOCKFILE={self.lock_file_path}\n")
            f.write(f'if [ ! -f "$LOCKFILE" ]; then\n')
            f.write(f"  touch $LOCKFILE\n")
            f.write(f"fi\n")
            # lock
            f.write(f"flock -x -n $LOCKFILE\n")
            f.write(f"{cmd}\n")
            # unlock
            f.write(f"flock -u $LOCKFILE\n")
            f.write(f"rm -f $LOCKFILE\n")

    def run(self, hparams_str: str) -> None:
        """Run the job with the given hyperparameters."""
        cmd = create_submit_command(
            self.platform,
            self.script_name,
            self.group,
            self.job_file_path,
            self.stdout_file_path,
            self.stderr_file_path,
            hparams_str,
        )
        print(f"Running the job with the command: `{cmd}`")
        cmds = cmd.split()
        self._start_time = time.time()
        if self.platform == "abci":
            _run2(cmds, self.lock_file_path)
        else:
            self._returncode = _run(
                cmds, self.stdout_file_path, self.stderr_file_path, self.timeout_seconds
            )
        self._end_time = time.time()

    def collect_result(self) -> str | None:
        """Collect the result of the job."""
        return _collect_result(self.stdout_file_path)

    def create_result_json(self, result: dict) -> None:
        """Create a json file to store the result of the job.
        The file name is `{job_name}.json`.
        """
        result_file_path = str(self.work_dir / f"{self.job_name}.json")
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

    def get_lock_file_path(self) -> str:
        """Get the path of the lock file."""
        return self.lock_file_path


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


def _run2(cmds: list[str], lock_file_path: str) -> None:
    """Run the job with the given hyperparameters.
    This function is for ABCI.
    """
    subprocess.run(cmds, capture_output=True, text=True)  # is run in the another node
    _wait_for_lock_file_creation(lock_file_path)
    _wait_for_unlock(lock_file_path)
    # finish the job
    return


def _wait_for_unlock(lock_file_path: str) -> None:
    """Wait until the lock file is unlocked."""
    if not Path(lock_file_path).exists():
        return

    while True:
        with open(lock_file_path, "r") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                time.sleep(0.01)


def _wait_for_lock_file_creation(lock_file_path: str) -> None:
    """Wait until the lock file is created."""
    while True:
        if Path(lock_file_path).exists():
            break
        time.sleep(0.01)


# def _collect_result(stdout_file: str) -> str | None:
#     """Collect the result of the job.

#     return:
#         The result of the job (objective value).
#     """
#     try:
#         with open(stdout_file, encoding="utf-8") as file:
#             lines = file.readlines()
#             if lines:
#                 return lines[-1].strip()
#             else:
#                 return None
#     except FileNotFoundError:
#         return None
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         return None


@retry(_MAX_NUM=60, _DELAY=1.0)
def _collect_result(stdout_file: str) -> str | None:
    """Collect the result of the job.

    return:
        The result of the job (objective value).

    Note:
        Retry reading the file if it is not found.
        This is because the file metadata may not be updated immediately after the job finishes.
    """
    with open(stdout_file, encoding="utf-8") as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
        else:
            raise ValueError("No result found.")
