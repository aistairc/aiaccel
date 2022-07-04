from pathlib import PosixPath
from pathlib import Path
import aiaccel
from aiaccel.util.retry import retry
from fasteners import InterProcessLock
from fasteners import ReaderWriterLock
import ast
import psutil
import shutil
from aiaccel.config import Config


class _base:
    def __init__(self, path: PosixPath, lock: PosixPath) -> None:
        self.path = path
        self.lock = lock
        self.rw_lock = ReaderWriterLock()

    def exists(self) -> None:
        return self.path.exists()

    def remove(self) -> None:
        if not self.exists():
            return

        if self.lock is None:
            self.path.unlink()
        else:
            with InterProcessLock(self.lock):
                self.path.unlink()

    def write(self, contents):
        if type(contents) != str:
            contents = str(contents)

        if self.lock is None:
            self.path.write_text(contents)
        else:
            with InterProcessLock(self.lock):
                with self.rw_lock.write_lock():
                    self.path.write_text(contents)

    def read(self):
        if not self.exists():
            return None

        if self.lock is None:
            text = self.path.read_text()
        else:
            with InterProcessLock(self.lock):
                with self.rw_lock.read_lock():
                    text = self.path.read_text()

        try:
            return ast.literal_eval(text)
        except ValueError:
            return(text)  # as string

    def copy(self, dst: PosixPath):
        if dst.exists():
            dst.unlink()

        if self.lock is None:
            shutil.copy2(self.path, dst)
        else:
            with InterProcessLock(self.lock):
                shutil.copy2(self.path, dst)

    def move(self, dst: PosixPath):
        if not self.exists():
            raise
        if dst.exists():
            raise
        self.copy(dst)
        self.remove()


class Datalist:
    def __init__(self):
        self.arr = []

    def last_index(self) -> int:
        return len(self.arr) - 1

    def clear(self) -> None:
        self.arr = []

    def add(self, index: int, object: _base) -> None:
        if index > self.last_index():
            for _ in range(index - self.last_index()):
                self.arr.append(None)
        self.arr[index] = object

    def get(self, index: int) -> any:
        if self.len == 0:
            return None
        if index > self.len - 1:
            return None
        if self.arr[index] is None:
            return None
        return self.arr[index].read()

    def set(self, index: int, set_data: any):
        self.arr[index].write(set_data)

    @property
    def len(self):
        return len(self.arr)

    @property
    def data(self):
        return self.arr

    def no_data(self) -> bool:
        return self.len == 0

    def all(self) -> list:
        return [d.read() for d in self.arr if d is not None]

    def all_delete(self) -> None:
        [d.remove() for d in self.arr if d is not None]


class _alive(_base):
    pass


class _pid(_base):
    def is_alive(self) -> bool:
        if not self.exits():
            return False
        return self.is_running(self.read())

    def is_running(self, pid: int) -> bool:
        """Check the process is running or not.

        Args:
            pid (int): A pid.

        Returns:
            bool: The process is running or not.
        """

        states = [
            "running",
            "sleeping",
            "disk-sleep",
            "stopped",
            "tracing-stop",
            "waking",
            "idle"
        ]
        try:
            p = psutil.Process(pid)
            return p.status() in states
        except psutil.NoSuchProcess:
            return False


class _error(_base):
    def __init__(self, config: Config, trial_id: int):
        self.config = config
        self.workspace = Path(self.config.workspace.get()).resolve()
        self.name_length = self.config.name_length.get()
        self.fmt = '%0{}d'.format(self.name_length)
        self.trial_id = self.zero_padding(trial_id)
        self.file_type = "yaml"
        self.file_name = f'{self.trial_id}.{self.file_type}'
        self.lock = self.workspace / aiaccel.dict_lock / f'err_{trial_id}'
        self.path = self.workspace / aiaccel.dict_error / self.file_name
        super().__init__(self.path, self.lock)

    def zero_padding(self, trial_id: int):
        return self.fmt % trial_id


class _result(_base):
    def __init__(self, config: Config, trial_id: int):
        self.config = config
        self.workspace = Path(self.config.workspace.get()).resolve()
        self.name_length = self.config.name_length.get()
        self.fmt = '%0{}d'.format(self.name_length)
        self.trial_id = self.zero_padding(trial_id)
        self.file_type = "result"
        self.file_name = f'{self.trial_id}.{self.file_type}'
        self.lock = self.workspace / aiaccel.dict_lock / f'res_{trial_id}'
        self.path = self.workspace / aiaccel.dict_result / self.file_name
        super().__init__(self.path, self.lock)

    def zero_padding(self, trial_id: int):
        return self.fmt % trial_id


class _timestamp(_base):
    def __init__(self, config: Config, trial_id: int):
        self.config = config
        self.workspace = Path(self.config.workspace.get()).resolve()
        self.name_length = self.config.name_length.get()
        self.fmt = '%0{}d'.format(self.name_length)
        self.trial_id = self.zero_padding(trial_id)
        self.file_type = "yaml"
        self.file_name = f'{self.trial_id}.{self.file_type}'
        self.lock = self.workspace / aiaccel.dict_lock / f'ts_{trial_id}'
        self.path = self.workspace / aiaccel.dict_timestamp / self.file_name
        super().__init__(self.path, self.lock)

    def zero_padding(self, trial_id: int):
        return self.fmt % trial_id


class _trial(_base):
    def __init__(self, config: Config, trial_id: int):
        self.config = config
        self.workspace = Path(self.config.workspace.get()).resolve()
        self.name_length = self.config.name_length.get()
        self.fmt = '%0{}d'.format(self.name_length)
        self.trial_id = self.zero_padding(trial_id)
        self.state = None
        self.lock = self.workspace / aiaccel.dict_lock / f'hp{trial_id}'
        self.file_type = 'hp'
        self.file_name = f'{self.trial_id}.{self.file_type}'

        self.path_ready = self.workspace / aiaccel.dict_hp_ready / self.file_name
        self.path_running = self.workspace / aiaccel.dict_hp_running / self.file_name
        self.path_finished = self.workspace / aiaccel.dict_hp_finished / self.file_name

        self.ready = _base(self.path_ready, self.lock)
        self.running = _base(self.path_running, self.lock)
        self.finished = _base(self.path_finished, self.lock)

    def zero_padding(self, trial_id: int):
        return self.fmt % trial_id

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def ready_to_running(self) -> None:
        self.ready.move(self.running.path)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def ready_to_finished(self) -> None:
        self.ready.move(self.finished.path)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def running_to_finished(self) -> None:
        self.running.move(self.finished.path)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def running_to_ready(self) -> None:
        self.running.move(self.ready.path)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def finished_to_ready(self) -> None:
        self.finished.move(self.ready.path)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def finished_to_running(self) -> None:
        self.finished.move(self.running.path)


class _hp(_base):
    def __init__(self, config: Config, trial_id: int):
        self.config = config
        self.workspace = Path(self.config.workspace.get()).resolve()
        self.name_length = self.config.name_length.get()
        self.fmt = '%0{}d'.format(self.name_length)
        self.trial_id = self.zero_padding(trial_id)
        self.file_type = 'hp'
        self.file_name = f'{self.trial_id}.{self.file_type}'
        self.lock = self.workspace / aiaccel.dict_lock / f'hp{trial_id}'
        self.path = self.workspace / aiaccel.dict_hp / self.file_name
        self.path_ready = self.workspace / aiaccel.dict_hp_ready / self.file_name
        super().__init__(self.path, self.lock)

    def zero_padding(self, trial_id: int):
        return self.fmt % trial_id

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def copy_to_ready(self):
        self.copy(self.path_ready)


class _jobstate(_base):
    def __init__(self, config: Config, trial_id: int):
        self.config = config
        self.workspace = Path(self.config.workspace.get()).resolve()
        self.name_length = self.config.name_length.get()
        self.fmt = '%0{}d'.format(self.name_length)
        self.trial_id = self.zero_padding(trial_id)
        self.file_type = 'txt'
        self.file_name = f'{self.trial_id}.{self.file_type}'
        self.lock = self.workspace / aiaccel.dict_lock / f'js{trial_id}'
        self.path = self.workspace / aiaccel.dict_jobstate / self.file_name
        super().__init__(self.path, self.lock)

    def zero_padding(self, trial_id: int):
        return self.fmt % trial_id
