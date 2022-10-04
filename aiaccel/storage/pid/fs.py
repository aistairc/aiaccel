# import psutil
from pathlib import Path
import aiaccel
from aiaccel.config import Config
from aiaccel.storage.model.fs import _pid


class Pid:
    def __init__(self, config: Config):
        self.workspace = Path(config.workspace.get()).resolve()
        self.name_length = config.name_length.get()
        self.dict_pid = self.workspace / aiaccel.dict_pid
        self.dict_lock = self.workspace / aiaccel.dict_lock

        file = {
            'optimizer': self.dict_pid / 'optimizer.yaml',
            'scheduler': self.dict_pid / 'scheduler.yaml'
        }

        lock = {
            'optimizer': self.dict_lock / 'pid_optimizer',
            'scheduler': self.dict_lock / 'pid_scheduler'
        }

        self.optimizer = _pid(file['optimizer'], lock['optimizer'])
        self.scheduler = _pid(file['scheduler'], lock['scheduler'])

    def _(self, module_name: str):
        if module_name == aiaccel.module_type_optimizer:
            return self.optimizer
        elif module_name == aiaccel.module_type_scheduler:
            return self.scheduler
        else:
            assert False

    def set_any_process_pid(self, module_name: str, pid: int) -> None:
        self._(module_name).write(pid)

    def get_any_process_pid(self, module_name: str) -> int:
        return self._(module_name).read()

    def all_delete(self):
        self.optimizer.remove()
        self.scheduler.remove()

    def delete_any_process_pid(self, module_name: str) -> None:
        self._(module_name).remove()
