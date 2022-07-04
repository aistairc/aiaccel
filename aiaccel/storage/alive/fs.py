from pathlib import PosixPath
from pathlib import Path
import aiaccel
from aiaccel.config import Config
from ..model.fs import _alive


class Alive:
    def __init__(self, config: Config):
        self.workspace = Path(config.workspace.get()).resolve()
        self.name_length = config.name_length.get()
        self.dict_alive = self.workspace / aiaccel.dict_alive
        self.dict_lock = self.workspace / aiaccel.dict_lock

        file = {
            'master': self.dict_alive / 'master.yaml',
            'optimizer': self.dict_alive / 'optimizer.yaml',
            'scheduler': self.dict_alive / 'scheduler.yaml'
        }

        lock = {
            'master': self.dict_lock / 'alive_master',
            'optimizer': self.dict_lock / 'alive_optimizer',
            'scheduler': self.dict_lock / 'alive_scheduler'
        }

        self.master = _alive(file['master'], lock['master'])
        self.optimizer = _alive(file['optimizer'], lock['optimizer'])
        self.scheduler = _alive(file['scheduler'], lock['scheduler'])
        self.contents = {}

    def _(self, process_name: str):
        if process_name == aiaccel.module_type_master:
            return self.master
        elif process_name == aiaccel.module_type_optimizer:
            return self.optimizer
        elif process_name == aiaccel.module_type_scheduler:
            return self.scheduler
        else:
            assert False

    def init_alive(self):
        pass

    def get_state(self) -> dict:
        alives = {
            'master': self.get_any_process_state('master'),
            'optimizer': self.get_any_process_state('optimizer'),
            'scheduler': self.get_any_process_state('scheduler')
        }
        return alives

    def set_any_process_state(self, process_name: str, alive_state: int) -> None:
        """Set the specified process state.

        Returns:
            None
        """
        if alive_state == 0:
            self._(process_name).remove()
        elif alive_state == 1:
            self._(process_name).write(self.contents)

    def check_alive(self, process_name: str) -> bool:
        alive = self.get_state()
        return True if alive[process_name] == 1 else False

    def get_any_process_state(self, process_name: str) -> int:
        alive_status = self._(process_name).exists()
        return int(alive_status)

    def stop_any_process(self, process_name: str) -> None:
        self._(process_name).remove()

    def all_delete(self):
        self.master.remove()
        self.optimizer.remove()
        self.scheduler.remove()
