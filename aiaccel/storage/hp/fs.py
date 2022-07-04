from pathlib import PosixPath
from pathlib import Path
from typing import Any
from typing import Union
import aiaccel
from aiaccel.storage.model.fs import _hp
from aiaccel.storage.model.fs import Datalist
from aiaccel.config import Config


class Hpdata:
    def __init__(
        self,
        trial_id: int,
        param_name=None,
        param_type=None,
        param_value=None
    ) -> None:
        self.trial_id = trial_id
        self.param_name = param_name
        self.param_type = param_type
        self.param_value = param_value

    def set(self, trial_id, param_name, param_type, param_value):
        self.trial_id = trial_id
        self.param_name = param_name
        self.param_type = param_type
        self.param_value = param_value


class Hp:
    def __init__(self, config: Config):
        self.config = config
        self.workspace = Path(config.workspace.get()).resolve()
        self.name_length = config.name_length.get()
        self.path = self.workspace / aiaccel.dict_hp
        # self.hps = [None]
        self.datas = Datalist()

    def add(self, trial_id: int):
        self.datas.add(
            trial_id,
            _hp(self.config, trial_id)
        )

    def write(self, trial_id: int, contents: any):
        self.datas.write(trial_id, contents)

    def clear(self):
        self.datas.clear()

    def update(self):
        self.clear()
        paths = sorted(list(self.path.glob("*.hp")))
        for path in paths:
            trial_id = int(path.stem)
            self.add(trial_id)

    def set_any_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value: Any,
        param_type: str
    ) -> None:
        """Set the specified parameter information for an any trial.

        Args:
            trial_id (int)    : Any trial id
            param_name (str)  : Hyperparameter name.
            param_value (Any) : Hyperparameter value
            param_type (str)  : Hyperparameter data type

        Returns:
            TrialTable | None
        """
        param = [
            {
                'trial_id': trial_id,
                'parameter_name': param_name,
                'value': param_value,
                'type': param_type
            }
        ]
        self.datas.clear()
        self.datas.add(trial_id, _hp(self.config, trial_id))
        self.datas.set(trial_id, param)
        self.datas.data[trial_id].copy_to_ready()

    def set_any_trial_params(self, trial_id: int, params: list) -> None:
        self.datas.clear()
        self.datas.add(trial_id, _hp(self.config, trial_id))
        self.datas.set(trial_id, params)
        self.datas.data[trial_id].copy_to_ready()

    def get_any_trial_params(self, trial_id: int) -> Union[None, list]:
        """ Obtain the set parameter information for any given trial.

        Args:
            trial_id(int): Any trial id.

        Returns:
            list[HpTable]
        """
        self.update()
        data = self.datas.get(trial_id)
        if data is None:
            return None

        hpdata = []
        for d in data:
            hpdata.append(
                Hpdata(
                    trial_id=trial_id,
                    param_name=d['parameter_name'],
                    param_type=d['type'],
                    param_value=d['value']
                )
            )
        return hpdata

    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        self.update()
        for hp in self.datas.data:
            hp.remove()
