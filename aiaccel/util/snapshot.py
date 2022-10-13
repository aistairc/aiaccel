import shutil
from pathlib import Path
import copy
from typing import Union
from typing import Any
from fasteners import InterProcessLock
from aiaccel.util import filesystem as fs
from aiaccel.util.retry import retry
import pickle


class SnapShot:
    def __init__(
        self,
        workspace: Path,
        process_name: str   # Assume 'scheduler' or 'optimizer' or 'master'
    ) -> None:

        self.process_name = process_name
        self.ws = workspace
        if type(self.ws) == str:
            self.ws = Path(workspace)
        self.ws = self.ws.resolve()
        self.lock = self.ws / 'lock' / f'snapshot_{self.process_name}'

    # @retry(_MAX_NUM=60, _DELAY=1.0)
    def save(self, trial_id: int, obj: Any) -> None:
        base_dir_path = self.ws / 'storage' / str(trial_id)
        if not base_dir_path.exists():
            base_dir_path.mkdir()

        pickle_file = base_dir_path / f"{self.process_name}.pickle"
        with InterProcessLock(self.lock):
            with open(pickle_file, 'wb') as f:
                pickle.dump(obj, f)

    # @retry(_MAX_NUM=60, _DELAY=1.0)
    def load(self, trial_id: int) -> Any:
        base_dir_path = self.ws / 'storage' / str(trial_id)

        pickle_file = base_dir_path / f"{self.process_name}.pickle"

        with InterProcessLock(self.lock):
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)

