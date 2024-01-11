from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from aiaccel.common import (
    dict_alive,
    dict_error,
    dict_finished,
    dict_hp,
    dict_jobstate,
    dict_lock,
    dict_log,
    dict_mpi,
    dict_output,
    dict_pid,
    dict_rank_log,
    dict_ready,
    dict_result,
    dict_runner,
    dict_running,
    dict_storage,
    dict_tensorboard,
    dict_timestamp,
    extension_hp,
)
from aiaccel.util import Suffix, load_yaml, make_directories


class Workspace:
    """Provides interface to workspace.

    Args:
        base_path (str): Path to the workspace.

    Attributes:
        path (Path): Path to the workspace.
        alive (Path): Path to "alive", i.e. `path`/alive.
        error (Path): Path to "error", i.e. 'path`/error.
        hp (Path): Path to "hp", i.e. `path`/hp.
        hp_ready (Path): Path to "ready", i.e. `path`/hp/ready.
        hp_running (Path): Path to "running", i.e. `path`/hp/running.
        hp_finished (Path): Path to "finished", i.e. `path`/hp/finished.
        jobstate (Path): Path to "jobstate", i.e. `path`/jobstate.
        lock (Path): Path to "lock", i.e. `path`/lock.
        log (Path): Path to "log", i.e. `path`/log.
        output (Path): Path to "abci_output", i.e. `path`/abci_output.
        pid (Path): Path to "pid", i.e. `path`/pid.
        result (Path): Path to "result", i.e. `path`/result.
        runner (Path): Path to "runner", i.e. `path`/runner.
        storage (Path): Path to "storage", i.e. `path`/storage.
        timestamp (Path): Path to "timestamp", i.e. `path`/timestamp.
        consists (list[Path]): A list of pathes under the workspace.
        results (Path): Path to the results which is prepared in the execution
            directory, i.e. "./results".

    """

    def __init__(self, base_path: str):
        self.path = Path(base_path).resolve()

        self.alive = self.path / dict_alive
        self.error = self.path / dict_error
        self.hp = self.path / dict_hp
        self.hp_ready = self.path / dict_hp / dict_ready
        self.hp_running = self.path / dict_hp / dict_running
        self.hp_finished = self.path / dict_hp / dict_finished
        self.jobstate = self.path / dict_jobstate
        self.lock = self.path / dict_lock
        self.log = self.path / dict_log
        self.mpi = self.path / dict_mpi
        self.rank_log = self.mpi / dict_rank_log
        self.output = self.path / dict_output
        self.pid = self.path / dict_pid
        self.result = self.path / dict_result
        self.runner = self.path / dict_runner
        self.storage = self.path / dict_storage
        self.tensorboard = self.path / dict_tensorboard
        self.timestamp = self.path / dict_timestamp
        self.consists = [
            self.alive,
            self.error,
            self.hp,
            self.hp_ready,
            self.hp_running,
            self.hp_finished,
            self.jobstate,
            self.lock,
            self.log,
            self.mpi,
            self.output,
            self.pid,
            self.rank_log,
            self.result,
            self.runner,
            self.storage,
            self.tensorboard,
            self.timestamp,
        ]
        self.results = Path("./results")
        self.retults_csv_file = self.path / "results.csv"
        self.final_result_file = self.path / "final_result.result"
        self.storage_file_path = self.storage / "storage.db"
        self.best_result_file = self.path / "best_result.yaml"

    def create(self) -> bool:
        """Create a work directory.

        Args:
            None

        Returns:
            None

        Raises:
            NotADirectoryError: It raises if a workspace argument (self.path)
                is not a directory.
        """
        if self.exists():
            return False

        make_directories(ds=self.consists, dict_lock=(self.lock))
        return True

    def exists(self) -> bool:
        """Returns whether workspace exists or not.

        Args:
            None

        Returns:
            bool: True if the workspace exists.
        """
        return self.path.exists()

    def clean(self) -> None:
        """Delete a workspace.

        It is assumed to be the first one to be executed.

        Args:
            None

        Returns:
            None
        """
        if not self.path.exists():
            return
        shutil.rmtree(self.path)
        return

    def check_consists(self) -> bool:
        """Check required directories exist or not.

        Args:
            None

        Returns:
            bool: All required directories exist or not.
        """
        for d in self.consists:
            if d.is_dir():
                continue
            else:
                return False
        return True

    def move_completed_data(self) -> Path | None:
        """Move workspace to under of results directory when finished.

        Args:
            None

        Returns:
            Path | None: Path of destination.

        Raises:
            FileExistsError: Occurs if destination directory already exists
                when the method is called.
        """

        dst = self.results / Suffix.date()
        if not self.results.exists():
            self.results.mkdir()

        if dst.exists():
            print(f"Destination directory already exists: {dst}")
            return None

        ignptn = shutil.ignore_patterns("*-journal")

        shutil.copytree(self.path, dst, ignore=ignptn)
        return dst

    def get_any_result_file_path(self, trial_id: int) -> Path:
        """Get result file path.

        Args:
            trial_id(int): Any trial id

        Returns:
            PosixPath: Path to result file.
        """
        return self.result / f"{trial_id}.{extension_hp}"

    def result_file_exists(self, trial_id: int) -> bool:
        """Check result file exists or not.

        Args:
            trial_id(int): Any trial id

        Returns:
            bool: True if result file exists.
        """
        path = self.get_any_result_file_path(trial_id)
        return path.exists()

    def get_any_trial_result(self, trial_id: int) -> dict[str, Any] | None:
        """Get any trial result.

        Args:
            trial_id(int): Any trial id

        Returns:
            dict: Trial result.
        """
        path = self.get_any_result_file_path(trial_id)
        if path.exists() is False:
            return None
        return load_yaml(path)

    def get_error_output_file(self, trial_id: int) -> Path:
        """ Get error output file path

        Args:
            trial_id(int): Any trial id

        Returns:
            Path: The path to the error output file.
        """
        return self.error / f"{trial_id}.txt"

    def get_runner_file(self, trial_id: int) -> Path:
        """
        Returns the file path for the runner script associated with the given trial ID.

        Args:
            trial_id(int): Any trial id

        Returns:
            Path: The file path for the runner script.
        """
        return self.runner / f"run_{trial_id}.sh"
