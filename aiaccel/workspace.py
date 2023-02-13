from __future__ import annotations
from pathlib import Path
import shutil

import aiaccel
from aiaccel.util import filesystem as fs
from aiaccel.util.retry import retry
from aiaccel.util.suffix import Suffix


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
        verification (Path): Path to "verification", i.e. `path`/verification.
        consists (list[Path]): A list of pathes under the workspace.
        results (Path): Path to the results which is prepared in the execution
            directory, i.e. "./results".

    """

    def __init__(self, base_path: str):
        self.path = Path(base_path).resolve()

        self.alive = self.path / aiaccel.dict_alive
        self.error = self.path / aiaccel.dict_error
        self.hp = self.path / aiaccel.dict_hp
        self.hp_ready = self.path / aiaccel.dict_hp / aiaccel.dict_ready
        self.hp_running = self.path / aiaccel.dict_hp / aiaccel.dict_running
        self.hp_finished = self.path / aiaccel.dict_hp / aiaccel.dict_finished
        self.jobstate = self.path / aiaccel.dict_jobstate
        self.lock = self.path / aiaccel.dict_lock
        self.log = self.path / aiaccel.dict_log
        self.output = self.path / aiaccel.dict_output
        self.pid = self.path / aiaccel.dict_pid
        self.result = self.path / aiaccel.dict_result
        self.runner = self.path / aiaccel.dict_runner
        self.storage = self.path / aiaccel.dict_storage
        self.timestamp = self.path / aiaccel.dict_timestamp
        self.verification = self.path / aiaccel.dict_verification

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
            self.output,
            self.pid,
            self.result,
            self.runner,
            self.storage,
            self.timestamp,
            self.verification
        ]
        self.results = Path("./results")

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

        fs.make_directories(
            ds=self.consists,
            dict_lock=(self.lock)
        )
        return True

    def exists(self) -> bool:
        """Returns whether workspace exists or not.

        Returns:
            bool: True if the workspace exists.
        """
        return self.path.exists()

    @retry(_MAX_NUM=300, _DELAY=1.0)
    def clean(self) -> None:
        """ Delete a workspace.

        It is assumed to be the first one to be executed.
        """
        if not self.path.exists():
            return
        shutil.rmtree(self.path)
        return

    @retry(_MAX_NUM=10, _DELAY=1.0)
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

    @retry(_MAX_NUM=10, _DELAY=1.0)
    def move_completed_data(self) -> Path | None:
        """ Move workspace to under of results directory when finished.

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
            raise FileExistsError

        ignptn = shutil.ignore_patterns('*-journal')

        shutil.copytree(self.path, dst, ignore=ignptn)
        return dst
