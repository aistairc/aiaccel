import shutil
import pathlib
from typing import Union
import aiaccel
from aiaccel.util import filesystem as fs
from aiaccel.util.retry import retry
from aiaccel.util.suffix import Suffix


class Workspace:
    def __init__(self, base_path: str):
        self.path = pathlib.Path(base_path).resolve()

        self.alive = self.path / aiaccel.dict_alive
        self.error = self.path / aiaccel.dict_error
        self.hp = self.path / aiaccel.dict_hp
        self.hp_new = self.path / aiaccel.dict_hp / aiaccel.dict_new
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
        self.ramdomstate = self.path / aiaccel.dict_randomstate
        self.serialize = self.path / aiaccel.dict_serialize
        self.state = self.path / aiaccel.dict_state
        self.storage = self.path / aiaccel.dict_storage
        self.snapshpt = self.path / aiaccel.dict_snapshot
        self.timestamp = self.path / aiaccel.dict_timestamp
        self.trial = self.path / aiaccel.dict_trial

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
            self.ramdomstate,
            self.serialize,
            self.state,
            self.storage,
            self.snapshpt,
            self.timestamp,
            self.trial,
            self.verification
        ]
        self.results = pathlib.Path("./results")

    def create(self) -> bool:
        """Create a work directory.

        Returns:
            None

        Raises:
            NotADirectoryError: It raises if a workspace argument (self.path) is
            not a directory.
        """
        if self.exists():
            return False

        fs.make_directories(
            ds=self.consists,
            dict_lock=(self.lock)
        )
        return True

    def exists(self):
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

    def move_completed_data(self) -> Union[None, pathlib.PosixPath]:
        """ Move workspace to under of results directory when finished.
        """

        dst = self.results / Suffix.date()
        if not self.results.exists():
            self.results.mkdir()

        if dst.exists():
            return

        shutil.copytree(self.path, dst)
        return dst

    # @property
    # def alive(self) -> PosixPath:
    #     return self.path / aiaccel.dict_alive

    # @property
    # def error(self) -> PosixPath:
    #     return self.path / aiaccel.dict_error

    # @property
    # def hp(self) -> PosixPath:
    #     return self.path / aiaccel.dict_hp

    # @property
    # def jobstate(self) -> PosixPath:
    #     return self.path / aiaccel.dict_jobstate

    # @property
    # def lock(self) -> PosixPath:
    #     return self.path / aiaccel.dict_lock

    # @property
    # def log(self) -> PosixPath:
    #     return self.path / aiaccel.dict_log

    # @property
    # def output(self) -> PosixPath:
    #     return self.path / aiaccel.dict_output

    # @property
    # def pid(self) -> PosixPath:
    #     return self.path / aiaccel.dict_pid

    # @property
    # def result(self) -> PosixPath:
    #     return self.path / aiaccel.dict_result

    # @property
    # def runner(self) -> PosixPath:
    #     return self.path / aiaccel.dict_runner

    # @property
    # def randomstate(self) -> PosixPath:
    #     return self.path / aiaccel.dict_randomstate

    # @property
    # def serialize(self) -> PosixPath:
    #     return self.path / aiaccel.dict_serialize

    # @property
    # def state(self) -> PosixPath:
    #     return self.path / aiaccel.dict_state

    # @property
    # def storage(self) -> PosixPath:
    #     return self.path / aiaccel.dict_storage

    # @property
    # def snapshpt(self) -> PosixPath:
    #     return self.path / aiaccel.dict_snapshot

    # @property
    # def timestamp(self) -> PosixPath:
    #     return self.path / aiaccel.dict_timestamp

    # @property
    # def trial(self) -> PosixPath:
    #     return self.path / aiaccel.dict_trial

    # @property
    # def ready(self) -> PosixPath:
    #     return self.path / aiaccel.dict_trial / aiaccel.dict_ready

    # @property
    # def running(self) -> PosixPath:
    #     return self.path / aiaccel.dict_trial / aiaccel.dict_running

    # @property
    # def finished(self) -> PosixPath:
    #     return self.path / aiaccel.dict_trial / aiaccel.dict_finished

    # @property
    # def verification(self) -> PosixPath:
    #     return self.path / aiaccel.dict_verification

    # @property
    # def results(self) -> PosixPath:
    #     return pathlib.Path("./results")
