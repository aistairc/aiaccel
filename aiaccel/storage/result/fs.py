from typing import Any
from typing import Union
from pathlib import Path
import aiaccel
from aiaccel.config import Config
from aiaccel.storage.model.fs import Datalist
from aiaccel.storage.model.fs import _result


class ResultTable:
    def __init__(self, trial_id=None, objective=None):
        self.trial_id = trial_id
        self.objective = objective


class Result:
    def __init__(self, config: Config):
        self.config = config
        self.workspace = Path(config.workspace.get()).resolve()
        self.path = self.workspace / aiaccel.dict_result
        self.file_type = "result"
        self.results = Datalist()

    def clear(self):
        self.results.clear()

    def add(self, trial_id: int):
        self.results.add(
            trial_id,
            _result(self.config, trial_id)
        )

    def set(self, trial_id: int, contents: dict) -> None:
        self.results.set(trial_id, contents)

    def get(self, trial_id: int) -> dict:
        return self.results.get(trial_id)

    def all(self, ignore_none=False) -> list:
        self.update()
        d = []
        if ignore_none:
            return [d.get() for d in self.results.data if d is not None]

        for data in self.results.data:
            if data is None:
                d.append(data)  # return [None]
            else:
                trial_id = data.read()['trial_id']
                objective = data.read()['result']
                d.append(ResultTable(trial_id=trial_id, objective=objective))
        return d

    def update(self):
        self.clear()
        paths = self.get_file_list()
        for path in paths:
            try:
                trial_id = int(path.stem)
                self.add(trial_id)
            except ValueError:
                pass
            finally:
                pass

    def get_file_list(self) -> list:
        return sorted(list(self.path.glob(f"*.{self.file_type}")))

    def set_any_trial_objective(self, trial_id: int, objective: Any) -> None:
        """Set any trial result value.

        Args:
            trial_id (int): Any trial id
            objective(Any):

        Returns:
            None
        """
        contents = {
            'trial_id': trial_id,
            'result': objective
        }
        self.update()
        self.add(trial_id)
        self.set(trial_id, contents)

    def get_any_trial_objective(self, trial_id) -> Union[None, int, float]:
        self.update()
        data = self.get(trial_id)
        if data is None:
            return None
        objective = data["result"]
        return objective

    def get_all_result(self) -> list:
        """Get all results

        Returns:
            Any
        """
        return self.all()

    def get_objectives(self) -> list:
        """Get all results in list.

        Returns:
            objectives(list): result values
        """
        data = self.get_all_result()
        return [d.objective for d in data]

    def get_bests(self, goal: str) -> list:
        """Obtains the sorted result.

        Returns:
            objectives(list): result values
        """
        goal = goal.lower()
        objectives = self.get_objectives()
        best_values = []

        if goal == "maximize":
            best_value = float("-inf")
            for objective in objectives:
                if best_value < objective:
                    best_value = objective
                best_values.append(best_value)
        elif goal == "minimize":
            best_value = float("inf")
            for objective in objectives:
                if best_value > objective:
                    best_value = objective
                best_values.append(best_value)
        else:
            return []
        return best_values

    def get_result_trial_id_list(self) -> Union[None, list]:
        """Obtains the sorted result.

        Returns:
            objectives(list): result values
        """
        data = self.get_all_result()
        return [d.trial_id for d in data if d is not None]

    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        self.update()
        self.results.all_delete()

    def delete_any_trial_objective(self, trial_id) -> None:
        self.update()
        self.results.remove(trial_id)
