from __future__ import annotations

import sys
import time

from pathlib import Path
from aiaccel.job.env import Abci, Local

import json

script_name = sys.argv[0]
__local__ = "local"  # for debug
__abci__ = "abci"


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

        self.job_file_path = str(self.work_dir / f"job{self.trial_id}.sh")
        self.stdout_file_path = str(self.work_dir / f"o{self.trial_id}")
        self.stderr_file_path = str(self.work_dir / f"e{self.trial_id}")

        if self.platform == __local__:
            self.excution_environment = Local(
                script_name,
                self.preamble,
                self.job_file_path,
                self.stdout_file_path,
                self.stderr_file_path,
            )
        elif self.platform == __abci__:
            self.excution_environment = Abci(
                script_name,
                self.preamble,
                self.group,
                self.job_file_path,
                self.work_dir,
                self.work_dir,
            )
        else:
            raise NotImplementedError(f"Platform '{self.platform}' not implemented.")

    def create(self, param: dict) -> None:
        """Create a executable file to run the job."""
        self.excution_environment.create(param)

    def run(self) -> None:
        """Run the job with the given hyperparameters."""
        self._start_time = time.time()
        self._returncode = self.excution_environment.run()
        self._end_time = time.time()

    def collect_result(self) -> str | None:
        """Collect the result of the job."""
        return self.excution_environment.collect_result()

    def create_result_json(self, result: dict) -> None:
        """Create a result file.
        retult: dict
        {
            "trial_id": int,
            "x": float,
            ...
            "objective_value": float,
        }
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

    def get_elapsed_time(self) -> float:
        """Get the elapsed time."""
        if self._start_time is None:
            raise RuntimeError("Job not started.")
        return time.time() - self._start_time
