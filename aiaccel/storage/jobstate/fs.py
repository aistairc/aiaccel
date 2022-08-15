from pathlib import Path
from typing import Union
import aiaccel
from aiaccel.config import Config
from aiaccel.storage.model.fs import _jobstate
from aiaccel.storage.model.fs import Datalist


class JobState:
    def __init__(self, config: Config):
        self.config = config
        self.workspace = Path(config.workspace.get()).resolve()
        self.file_hp_count_fmt = config.name_length.get()
        self.file_type = 'txt'
        self.path = self.workspace / aiaccel.dict_jobstate
        self.jobstates = Datalist()

    def get_file_list(self) -> list:
        return sorted(list(self.path.glob(f"*.{self.file_type}")))

    def add(self, trial_id: int):
        self.jobstates.add(trial_id, _jobstate(self.config, trial_id))

    def set(self, trial_id: int, contents: dict) -> None:
        self.jobstates.set(trial_id, contents)

    def update(self):
        self.jobstates.clear()
        paths = self.get_file_list()
        for path in paths:
            trial_id = int(path.stem)
            self.add(trial_id)

    def set_any_trial_jobstate(self, trial_id: int, state: str) -> None:
        self.update()
        self.add(trial_id)
        self.set(trial_id, state)

    def set_any_trial_jobstates(self, states: list) -> None:
        self.update()
        for state in states:
            self.add(state['trial_id'])
            self.set(state['trial_id'], state['jobstate'])

    def get_any_trial_jobstate(self, trial_id: int) -> Union[None, str]:
        self.update()
        return self.jobstates.get(trial_id)

    def get_all_trial_jobstate(self) -> list:
        self.update()
        return self.jobstates.all()

    def is_failure(self, trial_id: int) -> bool:
        self.update()
        state = self.jobstates.get(trial_id)
        return "failure" in state.lower()

    def all_delete(self) -> None:
        self.update()
        self.jobstates.all_delete()

    def delete_any_trial_jobstate(self, trial_id: int) -> None:
        self.update()
        self.jobstates.remove(trial_id)
