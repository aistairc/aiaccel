from __future__ import annotations

import threading
import importlib
from collections.abc import Callable
from pathlib import Path

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.aiaccel import Run


class PylocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer.

    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)

        self.run = None
        self.user_func = None

        self.user_func = self.get_callable_object(
            self.config.python_file.get(),
            self.config.function.get()
        )
        self.run = Run(self.config_path)

    def get_callable_object(self, file_path: str | Path, attr_name: str
                            ) -> Callable[[dict], float]:
        """ Loads the specified module from the specified python program.

        Args:
            file_path (str, pathlib.Path): A user program file path (python
            file only).
            attr_name (str): A name of objective function in user program.

        Returns:
            Callable[[dict], float]:
        """
        spec = importlib.util.spec_from_file_location("user_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, attr_name)

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """

        trial_ids = self.storage.trial.get_ready()
        if trial_ids is None or len(trial_ids) == 0:
            return True

        for trial_id in trial_ids:
            self._serialize(trial_id)
            if self.num_node > 1:
                th = threading.Thread(target=self.execute, args=(trial_id,))
                th.start()
            else:
                self.execute(trial_id)

        return True

    def get_any_trial_xs(self, trial_id: int) -> dict | None:
        """Gets a parameter list of specific trial ID from Storage object.

        Args:
            trial_id (int): Trial ID.

        Returns:
            dict | None: A dictionary of parameters. None if the parameter
                specified by the given trial ID is not registered.
        """
        params = self.storage.hp.get_any_trial_params(trial_id=trial_id)
        if params is None:
            return {}

        xs = {}
        for param in params:
            xs[param.param_name] = param.param_value

        return xs

    def execute(self, trial_id: int) -> None:
        """ Executes the loaded callable object.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """
        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='running')

        xs = self.get_any_trial_xs(trial_id)
        _, y, err, start_time, end_time = self.run.execute(self.user_func, xs, y_data_type=None)
        self.report(trial_id, y, err, start_time, end_time)

        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')
        self.create_result_file(trial_id)

        return

    def report(
        self, trial_id: int, y: any, err: str, start_time: str,
        end_time: str
    ) -> None:
        """Saves results in the Storage object.

        Args:
            trial_id (int): Trial ID.
            xs (dict): A dictionary of parameters.
            y (any): Objective value.
            err (str): Error string.
            start_time (str): Execution start time.
            end_time (str): Execution end time.
        """

        self.storage.result.set_any_trial_objective(trial_id, y)
        self.storage.timestamp.set_any_trial_start_time(trial_id, start_time)
        self.storage.timestamp.set_any_trial_end_time(trial_id, end_time)
        if err != "":
            self.storage.error.set_any_trial_error(trial_id, err)

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['run']
        del obj['user_func']
        return obj
