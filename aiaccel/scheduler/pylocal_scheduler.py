import threading
import importlib

from typing import Union
from pathlib import Path

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.aiaccel import Run
from aiaccel.util.time_tools import get_time_now


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

    def get_callable_object(self, file_path: Union[str, Path], attr_name: str) -> callable:
        """ Loads the specified module from the specified python program.

        Args:
            file_path (str, pathlib.Path): A user program file path.(python file only)
            attr_name (str): A name of objective function in user program.

        Returns:
            callable
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

    def execute(self, trial_id: int) -> None:
        """ Executes the loaded callable object.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """
        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='running')

        start_time = get_time_now()
        xs, y, err = self.run.execute(self.user_func, trial_id, y_data_type=None)
        end_time = get_time_now()
        self.run.report(trial_id, xs, y, err, start_time, end_time)

        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')
        self.create_result_file(trial_id)

        return

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['run']
        del obj['user_func']
        return obj
