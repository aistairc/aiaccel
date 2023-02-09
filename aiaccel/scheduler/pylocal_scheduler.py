from __future__ import annotations

from multiprocessing import Pool
import importlib
from collections.abc import Callable
from pathlib import Path

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.aiaccel import Run
from aiaccel.util.aiaccel import WrapperInterface
from aiaccel.util.aiaccel import set_logging_basicConfig
from aiaccel.util.time_tools import get_time_now
from aiaccel.util.cast import cast_y

from aiaccel.config import Config


class PylocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer.

    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.pool = Pool(self.num_node, initializer=initializer(self.config_path))
        self.run = Run(self.config_path)
        self.user_func = get_callable_object(
            self.config.python_file.get(),
            self.config.function.get()
        )
        self.workspace = Path(self.config.workspace.get()).resolve()
        self.com = WrapperInterface()

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
            if self.num_node <= 1:
                self.execute(trial_id)

        if self.num_node > 1:
            args = []
            for trial_id in trial_ids:
                self.storage.trial.set_any_trial_state(trial_id=trial_id, state='running')
                xs = self.run.get_any_trial_xs(trial_id)
                args.append([trial_id, xs])
            for trial_id, xs, y, err, start_time, end_time in self.pool.imap_unordered(self.execute_wrapper, args):
                self.run.report(trial_id, xs, y, err, start_time, end_time)
                self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')
                self.create_result_file(trial_id)

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

    def execute_wrapper(self, args) -> None:
        trial_id, xs = args
        start_time = get_time_now()

        set_logging_basicConfig(self.workspace, trial_id)
        y = None
        err = ""

        try:
            y = cast_y(user_func(xs), y_data_type=None)
        except BaseException as e:
            err = str(e)
        finally:
            self.com.out(objective_y=y, objective_err=err)

        end_time = get_time_now()

        return trial_id, xs, y, err, start_time, end_time

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['run']
        del obj['user_func']
        del obj['pool']
        return obj


def initializer(config_path: str | Path):
    # Redefinition of variables to be removed by pickle conversion
    global user_func
    config = Config(config_path)
    user_func = get_callable_object(
        config.python_file.get(),
        config.function.get()
    )


def get_callable_object(file_path: str | Path, attr_name: str
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
