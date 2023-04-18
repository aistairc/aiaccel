from __future__ import annotations
from aiaccel.util.cast import cast_y
from aiaccel.util.aiaccel import Run, set_logging_file_for_trial_id
from aiaccel.util import get_time_now
from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from omegaconf.dictconfig import DictConfig
from aiaccel.config import load_config
from typing import Any, List
from subprocess import Popen

from importlib.util import module_from_spec, spec_from_file_location
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path


# These are for avoiding mypy-errors from initializer().
# `global` does not work well.
# https://github.com/python/mypy/issues/5732
user_func: Any
workspace: Path


class PylocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.run = Run(self.config.config_path)
        self.processes: List[Any] = []

        Pool_ = Pool if self.num_node > 1 else ThreadPool
        self.pool = Pool_(self.num_node, initializer=initializer,
                          initargs=(self.config.config_path,))

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
            self.storage.trial.set_any_trial_state(
                trial_id=trial_id, state="running")
            args.append([trial_id, self.get_any_trial_xs(trial_id)])
            self._serialize(trial_id)

        for trial_id, xs, ys, err, start_time, end_time in self.pool.imap_unordered(execute, args):
            self.report(trial_id, ys, err, start_time, end_time)
            self.storage.trial.set_any_trial_state(
                trial_id=trial_id, state="finished")

            self.create_result_file(
                trial_id, xs, ys, err, start_time, end_time)

        return True

    def post_process(self) -> None:
        for process in self.processes:
            process.wait()

        super().post_process()

    def get_any_trial_xs(self, trial_id: int) -> dict[str, Any] | None:
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

    def report(self, trial_id: int, ys: list[Any], err: str, start_time: str, end_time: str) -> None:
        """Saves results in the Storage object.

        Args:
            trial_id (int): Trial ID.
            xs (dict): A dictionary of parameters.
            y (Any): Objective value.
            err (str): Error string.
            start_time (str): Execution start time.
            end_time (str): Execution end time.
        """

        self.storage.result.set_any_trial_objective(trial_id, ys)
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

    def get_result_file_path(self) -> Path:
        """Get a path to the result file.

        Args:
            trial_id (int): Trial Id.

        Returns:
            PosixPath: A Path object which points to the result file.
        """
        return self.workspace.get_any_result_file_path(self.trial_id.get())

    def create_result_file(
        self, trial_id: int, xs: dict[str, Any], ys: list[Any], error: str, start_time: str, end_time: str
    ) -> None:
        args = {
            "file": self.workspace.get_any_result_file_path(trial_id),
            "trial_id": str(trial_id),
            "config": self.config.config_path,
            "start_time": start_time,
            "end_time": end_time,
            "error": error,
        }

        if len(error) == 0:
            del args["error"]

        commands = ["aiaccel-set-result"]
        for key in args.keys():
            commands.append("--" + key)
            commands.append(str(args[key]))

        commands.append("--objective")
        for y in ys:
            commands.append(str(y))

        for key in xs.keys():
            commands.append("--" + key)
            commands.append(str(xs[key]))

        self.processes.append(Popen(commands))

        return None

    def __getstate__(self) -> dict[str, Any]:
        obj = super().__getstate__()
        del obj["run"]
        del obj["pool"]
        del obj["processes"]
        return obj


def initializer(config_path: str | Path) -> None:
    """Initializer for multiprocessing.Pool.

    Args:
        config_path (str | Path): Path to the configuration file.
    Returns:
        None
    """
    global user_func, workspace

    config = load_config(config_path)

    # Load the specified module from the specified python program.
    spec = spec_from_file_location("user_module", config.generic.python_file)
    if spec is None:
        raise ValueError("Invalid python_path.")
    module = module_from_spec(spec)
    if spec.loader is None:
        raise ValueError("spec.loader not defined.")
    spec.loader.exec_module(module)

    user_func = getattr(module, config.generic.function)

    workspace = Path(config.generic.workspace).resolve()


def execute(args: Any) -> tuple[int, dict[str, Any], list[Any], str, str, str]:
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
        # y = cast_y(user_func(xs), y_data_type=None)
        y = user_func(xs)
        if isinstance(y, list):
            y = [cast_y(yi, y_data_type=None) for yi in y]
        else:
            y = [cast_y(y, y_data_type=None)]
    except BaseException as e:
        err = str(e)
        y = [None]
    else:
        err = ""

    end_time = get_time_now()

    return trial_id, xs, y, err, start_time, end_time
