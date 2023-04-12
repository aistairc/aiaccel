from __future__ import annotations

from pathlib import Path
from typing import Any

from aiaccel.storage import (Error, Hp, JobState, Result, Serializer,
                             TimeStamp, Trial)


class Storage:
    """Database
    """

    def __init__(self, _db_path: Path | str) -> None:
        self.db_path = Path(_db_path)
        self.trial = Trial(self.db_path)
        self.hp = Hp(self.db_path)
        self.result = Result(self.db_path)
        self.jobstate = JobState(self.db_path)
        self.error = Error(self.db_path)
        self.timestamp = TimeStamp(self.db_path)
        self.variable = Serializer(self.db_path)

    def current_max_trial_number(self) -> int | None:
        """Get the current maximum number of trials.

        Returns:
            trial_id (int): Any trial id

        Todo:
            Refuctoring
        """

        trial_ids = self.trial.get_all_trial_id()
        if trial_ids is None or len(trial_ids) == 0:
            return None

        return max(trial_ids)

    def get_ready(self) -> list[Any]:
        """Get a trial number for the ready state.

        Returns:
            trial_ids (list): trial ids in ready states
        """
        return self.trial.get_ready()

    def get_running(self) -> list[Any]:
        """Get a trial number for the running state.

        Returns:
            trial_ids (list): trial ids in running states
        """
        return self.trial.get_running()

    def get_finished(self) -> list[Any]:
        """Get a trial number for the finished state.

        Returns:
            trial_ids (list): trial ids in finished states
        """
        return self.trial.get_finished()

    def get_num_ready(self) -> int:
        """Get the number of trials in the ready state.

        Returns:
            int: number of ready state in trials
        """
        return len(self.trial.get_ready())

    def get_num_running(self) -> int:
        """Get the number of trials in the running state.

        Returns:
            int: number of running state in trials
        """
        return len(self.trial.get_running())

    def get_num_finished(self) -> int:
        """Get the number of trials in the finished state.

        Returns:
            int: number of finished state in trials
        """
        return len(self.trial.get_finished())

    def is_ready(self, trial_id: int) -> bool:
        """Whether the specified trial ID is ready or not.

        Args:
            trial_id (int): Any trial id

        Returns:
            bool
        """
        return trial_id in self.trial.get_ready()

    def is_running(self, trial_id: int) -> bool:
        """Whether the specified trial ID is running or not.

        Args:
            trial_id (int): Any trial id

        Returns:
            bool
        """
        return trial_id in self.trial.get_running()

    def is_finished(self, trial_id: int) -> bool:
        """Whether the specified trial ID is finished or not.

        Args:
            trial_id (int): Any trial id

        Returns:
            bool
        """
        return trial_id in self.trial.get_finished()

    def get_hp_dict(self, trial_id: Any) -> Any:
        """Obtain information on a specified trial in dict.

        Args:
            trial_id_str(str): trial id

        Returns:
            dict | None: Any trials information
        """

        data = self.hp.get_any_trial_params(trial_id=trial_id)
        if data is None:
            return None

        hp = []
        for d in data:
            param_name = d.param_name
            dtype = d.param_type  # str
            value = d.param_value

            if dtype.lower() == "float":
                value = float(d.param_value)
            elif dtype.lower() == "int":
                value = int(float(d.param_value))
            elif dtype.lower() == "categorical":
                value == str(d.param_value)
            else:  # pragma: no cover
                pass  # not reached

            hp.append(
                {
                    'parameter_name': param_name,
                    'type': dtype,
                    'value': value
                }
            )
        result = self.result.get_any_trial_objective(trial_id=trial_id)
        start_time = self.timestamp.get_any_trial_start_time(trial_id=trial_id)
        end_time = self.timestamp.get_any_trial_end_time(trial_id=trial_id)
        error = self.error.get_any_trial_error(trial_id=trial_id)

        content: dict[str, str | int | float | list[Any]] = {}
        content['trial_id'] = trial_id
        content['parameters'] = hp
        content['result'] = result
        content['start_time'] = start_time
        content['end_time'] = end_time

        if error is not None and len(error) > 0:
            content['error'] = error

        return content

    def get_best_trial(self, goals: list[str]) -> tuple[list[int] | None, list[float] | None]:
        """Get best trial number and best value.

        Args:
            goals(list[str]): minimize | maximize

        Returns:
            best(tuple): (trial_id, value)
        """

        best_values = []
        for i in range(len(goals)):
            if goals[i].lower() == 'maximize':
                best_values.append(float('-inf'))
            elif goals[i].lower() == 'minimize':
                best_values.append(float('inf'))
            else:
                return None, None

        best_trial_id = 0
        best_trial_ids = [0] * len(goals)

        results_d = self.result.get_all_result()
        for trial_id in results_d.keys():
            value = results_d[trial_id]

            if isinstance(value, float):
                values = [value]
            else:
                values = value

            if len(values) != len(goals):
                return None, None

            for i, val in enumerate(values):
                try:
                    float(val)
                except (ValueError, TypeError):
                    return None, None

                best_value = best_values[i]
                best_trial_id = best_trial_ids[i]

                if goals[i].lower() == 'maximize':
                    if best_value < val:
                        best_value = val
                        best_trial_id = trial_id

                elif goals[i].lower() == 'minimize':
                    if best_value > val:
                        best_value = val
                        best_trial_id = trial_id

                best_values[i] = best_value
                best_trial_ids[i] = best_trial_id

        return best_trial_ids, best_values

    def get_best_trial_dict(self, goals: list[str]) -> list[Any] | None:
        """Get best trial information in dict format.

        Args:
            goals(list[str]): minimize | maximize

        Returns:
            -(dict): Any trials information
        """
        best_trial_ids, _ = self.get_best_trial(goals)
        if best_trial_ids is None:
            return None

        hps = []
        for trial_id in best_trial_ids:
            hps.append(self.get_hp_dict(trial_id))
        return hps

    def get_result_and_error(self, trial_id: int) -> tuple[Any, Any]:
        """Get results and errors for a given trial number.

        Args:
            trial_id (int): Any trial id

        Returns:
            tuple(result, error)
        """
        r = self.result.get_any_trial_objective(trial_id=trial_id)
        e = self.error.get_any_trial_error(trial_id=trial_id)
        return (r, e)

    def delete_trial_data_after_this(self, trial_id: int) -> None:
        max_trial_id = self.current_max_trial_number()
        if max_trial_id is not None:
            for i in range(trial_id + 1, max_trial_id + 1):
                self.delete_trial(i)

    def delete_trial(self, trial_id: int) -> None:
        self.error.delete_any_trial_error(trial_id)
        self.jobstate.delete_any_trial_jobstate(trial_id)
        self.result.delete_any_trial_objective(trial_id)
        self.variable.delete_any_trial_variable(trial_id)
        self.timestamp.delete_any_trial_timestamp(trial_id)
        self.trial.delete_any_trial_state(trial_id)
        self.hp.delete_any_trial_params(trial_id)

    def rollback_to_ready(self, trial_id: int) -> None:
        if self.hp.get_any_trial_params(trial_id) is None:
            self.delete_trial(trial_id)
            return
        self.error.delete_any_trial_error(trial_id)
        self.jobstate.delete_any_trial_jobstate(trial_id)
        self.result.delete_any_trial_objective(trial_id)
        self.hp.delete_any_trial_params(trial_id)
        self.trial.delete_any_trial_state(trial_id)
