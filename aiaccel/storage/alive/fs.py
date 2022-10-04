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
            'optimizer': self.dict_alive / 'optimizer.yaml',
            'scheduler': self.dict_alive / 'scheduler.yaml'
        }

        lock = {
            'optimizer': self.dict_lock / 'alive_optimizer',
            'scheduler': self.dict_lock / 'alive_scheduler'
        }

        self.optimizer = _alive(file['optimizer'], lock['optimizer'])
        self.scheduler = _alive(file['scheduler'], lock['scheduler'])
        self.contents = {}

    def _(self, module_name: str):
        if module_name == aiaccel.module_type_optimizer:
            return self.optimizer
        elif module_name == aiaccel.module_type_scheduler:
            return self.scheduler
        else:
            assert False

    def init_alive(self):
        pass

    def get_state(self) -> dict:
        alives = {
            'optimizer': self.get_any_process_state('optimizer'),
            'scheduler': self.get_any_process_state('scheduler')
        }
        return alives

    def set_any_process_state(self, module_name: str, alive_state: int) -> None:
        """Set the specified process state.

        Returns:
            None
        """
        if alive_state == 0:
            self._(module_name).remove()
        elif alive_state == 1:
            self._(module_name).write(self.contents)

    def check_alive(self, module_name: str) -> bool:
        alive = self.get_state()
        return True if alive[module_name] == 1 else False

    def get_any_process_state(self, module_name: str) -> int:
        alive_status = self._(module_name).exists()
        return int(alive_status)

    def stop_any_process(self, module_name: str) -> None:
        self._(module_name).remove()

    def all_delete(self):
        self.optimizer.remove()
        self.scheduler.remove()
