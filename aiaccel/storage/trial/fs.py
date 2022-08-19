from pathlib import Path
from typing import Union
import aiaccel
from aiaccel.storage.model.fs import _trial
from aiaccel.storage.model.fs import Datalist
from aiaccel.config import Config


class Trial:
    def __init__(self, config: Config):
        self.config = config
        self.workspace = Path(config.workspace.get()).resolve()
        self.name_length = config.name_length.get()
        self.path = self.workspace / aiaccel.dict_hp
        self.dict_ready = self.workspace / aiaccel.dict_hp_ready
        self.dict_running = self.workspace / aiaccel.dict_hp_running
        self.dict_finished = self.workspace / aiaccel.dict_hp_finished

        if not self.dict_ready.exists():
            raise
        if not self.dict_running.exists():
            raise
        if not self.dict_finished.exists():
            raise

        self.trials = Datalist()

    def add(self, trial_id) -> None:
        self.trials.add(
            trial_id,
            _trial(self.config, trial_id)
        )

    def update(self):
        self.trials.clear()
        paths = sorted(list(self.path.glob("*.hp")))
        for path in paths:
            trial_id = int(path.stem)
            self.add(trial_id)

    def ready_to_running(self, trial_id: int):
        self.update()
        self.trials.data[trial_id].ready_to_running()

    def ready_to_finished(self, trial_id: int):
        self.update()
        self.trials.data[trial_id].ready_to_finished()

    def running_to_finished(self, trial_id: int):
        self.update()
        self.trials.data[trial_id].running_to_finished()

    def running_to_ready(self, trial_id: int):
        self.update()
        self.trials.data[trial_id].running_to_ready()

    def finished_to_ready(self, trial_id: int):
        self.update()
        self.trials.data[trial_id].finished_to_ready()

    def finished_to_running(self, trial_id: int):
        self.update()
        self.trials.data[trial_id].finished_to_running()

    def get_ready(self) -> list:
        readies = [
            int(Path(elem).stem)
            for elem in sorted(
                list(self.dict_ready.glob("*.hp"))
            )
        ]
        return readies

    def get_running(self) -> list:
        runnings = [
            int(Path(elem).stem)
            for elem in sorted(
                list(self.dict_running.glob("*.hp"))
            )
        ]
        return runnings

    def get_finished(self) -> list:
        finisheds = [
            int(Path(elem).stem)
            for elem in sorted(
                list(self.dict_finished.glob("*.hp"))
            )
        ]
        return finisheds

    def get_num_ready(self) -> int:
        return len(self.get_ready())

    def get_num_running(self) -> int:
        return len(self.get_running())

    def get_num_finished(self) -> int:
        return len(self.get_finished())

    def get_num_of_all_hp_files(self, src) -> list:
        n = self.get_num_ready()
        n += self.get_num_running()
        n += self.get_num_finished()
        return n

    # def get_any_trial(self, trial_id: int) -> None:
    #     assert False

    def get_any_trial_state(self, trial_id: int) -> str:
        """Get any trials state.

        Args:
            trial_id (int): Any trial id

        Returns:
            trials state(str): ready, running, finished
        """
        if trial_id in self.get_ready():
            return 'ready'
        elif trial_id in self.get_running():
            return 'running'
        elif trial_id in self.get_finished():
            return 'finished'
        else:
            return None

    def get_any_state_list(self, state: str) -> Union[None, list]:
        """Get any trials numbers.

        Args:
            trials state(str): ready, running, finished

        Returns:
            trial ids(list[int])
        """
        if state == 'ready':
            return self.get_ready()
        elif state == 'running':
            return self.get_running()
        elif state == 'finished':
            return self.get_finished()
        else:
            assert False

    def set_any_trial_state(self, trial_id: int, state: str) -> None:
        """Set any trials numbers.

        Args:
            trial_id (int): Any trial id
            trials state(str): ready, running, finished

        Returns:
            None
        """
        now_state = self.get_any_trial_state(trial_id=trial_id)
        if now_state == "ready":
            if state == "ready":
                pass
            elif state == 'running':
                self.ready_to_running(trial_id)
            elif state == 'finished':
                self.ready_to_finished(trial_id)
            else:
                assert False

        elif now_state == "running":
            if state == "ready":
                self.running_to_ready(trial_id)
            elif state == 'running':
                pass
            elif state == 'finished':
                self.running_to_finished(trial_id)
            else:
                assert False

        elif now_state == "finished":
            if state == "ready":
                self.finished_to_ready(trial_id)
            elif state == "running":
                self.finished_to_running(trial_id)
            elif state == "finished":
                pass
            else:
                assert False

    def get_all_trial_id(self) -> list:
        """
        Returns:
            trial ids(list[int])
        """
        return self.get_ready() + self.get_running() + self.get_finished()

    def all_delete(self):
        self.update()
        self.trials.all_delete()

    def delete_any_trial_state(self, trial_id: int) -> None:
        self.update()
        self.trials.data[trial_id].delete()
