from __future__ import annotations

from multiprocessing.pool import Pool, ThreadPool
import importlib
from pathlib import Path

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.aiaccel import Run
from aiaccel.util.aiaccel import WrapperInterface
from aiaccel.util.aiaccel import set_logging_file_for_trial_id
from aiaccel.util.time_tools import get_time_now
from aiaccel.util.cast import cast_y
from aiaccel.config import Config


class PylocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer.

    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.run = Run(self.config_path)
        self.com = WrapperInterface()

        Pool_ = Pool if self.num_node > 1 else ThreadPool
        self.pool = Pool_(self.num_node, initializer=initializer, initargs=(self.config_path,))

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """

        trial_ids = self.storage.trial.get_ready()
        if trial_ids is None or len(trial_ids) == 0:
            return True

        args = []
        for trial_id in trial_ids:
            self.storage.trial.set_any_trial_state(trial_id=trial_id, state='running')
            args.append([trial_id, self.get_any_trial_xs(trial_id)])
            self._serialize(trial_id)

        for trial_id, _, y, err, start_time, end_time in self.pool.imap_unordered(execute, args):
            self.com.out(objective_y=y, objective_err=err)
            self.report(trial_id, y, err, start_time, end_time)
            self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')

            self.create_result_file(trial_id)

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

    def create_model(self) -> None:
        """Creates model object of state machine.
        Returns:
            None: Because it does not use the state transition model.
        """
        return None

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['run']
        del obj['pool']
        return obj


def initializer(config_path: str | Path) -> None:
    """Initializer for multiprocessing.Pool.

    Args:
        config_path (str | Path): Path to the configuration file.
    Returns:
        None
    """
    global user_func, workspace

    config = Config(config_path)

    # Load the specified module from the specified python program.
    spec = importlib.util.spec_from_file_location("user_module", config.python_file.get())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    user_func = getattr(module, config.function.get())

    workspace = Path(config.workspace.get()).resolve()


def execute(args: list) -> tuple:
    """Executes the specified function with the specified arguments.

    Args:
        args (list): Arguments.
    Returns:
        tuple: Trial ID, arguments, objective value, error string, start time, end time.
    """
    trial_id, xs = args

    start_time = get_time_now()
    set_logging_file_for_trial_id(workspace, trial_id)

    try:
        y = cast_y(user_func(xs), y_data_type=None)
    except BaseException as e:
        err = str(e)
        y = None
    else:
        err = ""

    end_time = get_time_now()

    return trial_id, xs, y, err, start_time, end_time
