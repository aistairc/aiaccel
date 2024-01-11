from __future__ import annotations

import shutil
from pathlib import Path

from aiaccel.common import (
    dict_error,
    dict_lock,
    dict_log,
    dict_mpi,
    dict_rank_log,
    dict_runner,
    dict_stderr,
    dict_stdout,
    dict_tensorboard,
    file_best_result,
    file_final_result,
    file_result_csv,
    file_storage,
)
from aiaccel.util import Suffix, make_directories


class Workspace:
    """Provides interface to workspace.

    Args:
        base_path (str): Path to the workspace.

    Attributes:
        path (Path): Path to the workspace.
        error (Path): Path to the error directory.
        lock (Path): Path to the lock directory.
        log (Path): Path to the log directory.
        mpi (Path): Path to the mpi directory.
        rank_log (Path): Path to the rank_log directory.
        stderr (Path): Path to the stderr directory.
        stdout (Path): Path to the stdout directory.
        runner (Path): Path to the runner directory.
        tensorboard (Path): Path to the tensorboard directory.
        consists (list[Path]): List of required directories.
        results (Path): Path to the results directory.
        result_csv_file (Path): Path to the result.csv file.
        final_result_file (Path): Path to the final_result.yaml file.
        storage_file_path (Path): Path to the storage.db file.
        best_result_file (Path): Path to the best_result.yaml file.
    """

    def __init__(self, base_path: str):
        self.path = Path(base_path).resolve()

        self.error = self.path / dict_error
        self.lock = self.path / dict_lock
        self.log = self.path / dict_log
        self.mpi = self.path / dict_mpi
        self.rank_log = self.mpi / dict_rank_log
        self.stderr = self.path / dict_stderr
        self.stdout = self.path / dict_stdout
        self.runner = self.path / dict_runner
        self.tensorboard = self.path / dict_tensorboard
        self.consists = [
            self.error,
            self.lock,
            self.log,
            self.mpi,
            self.stdout,
            self.stderr,
            self.rank_log,
            self.runner,
            self.tensorboard,
        ]
        self.results = Path("./results")
        self.result_csv_file = self.path / file_result_csv
        self.final_result_file = self.path / file_final_result
        self.storage_file_path = self.path / file_storage
        self.best_result_file = self.path / file_best_result

    def create(self) -> bool:
        """Create a work directory.

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

        Returns:
            bool: True if the workspace exists.
        """
        return self.path.exists()

    def clean(self) -> None:
        """Delete a workspace.

        It is assumed to be the first one to be executed.
        """
        if not self.path.exists():
            return
        shutil.rmtree(self.path)
        return

    def check_consists(self) -> bool:
        """Check required directories exist or not.

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

        Raises:
            FileExistsError: Occurs if destination directory already exists
                when the method is called.

        Returns:
            Path | None: Path of destination.
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

    def get_error_output_file(self, trial_id: int) -> Path:
        return self.error / f"{trial_id}.txt"

    def get_runner_file(self, trial_id: int) -> Path:
        return self.runner / f"run_{trial_id}.sh"
