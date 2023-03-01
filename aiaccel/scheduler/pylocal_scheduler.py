from __future__ import annotations

import importlib
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

from omegaconf.dictconfig import DictConfig

from aiaccel.config import load_config
from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.aiaccel import (Run, WrapperInterface,
                                  set_logging_file_for_trial_id)
from aiaccel.util.cast import cast_y
from aiaccel.util.time_tools import get_time_now


class PylocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer.

    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.run = Run(self.config.config_path)
        self.com = WrapperInterface()

        Pool_ = Pool if self.num_node > 1 else ThreadPool
        self.pool = Pool_(self.num_node, initializer=initializer, initargs=(self.config.config_path,))

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
            args.append([trial_id, self.run.get_any_trial_xs(trial_id)])
            self._serialize(trial_id)

        for trial_id, xs, y, err, start_time, end_time in self.pool.imap_unordered(execute, args):
            self.com.out(objective_y=y, objective_err=err)
            self.run.report(trial_id, xs, y, err, start_time, end_time)
            self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')

            self.create_result_file(trial_id)

        return True

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['run']
        del obj['pool']
        return obj

    def create_model(self) -> None:
        """Creates model object of state machine.

        Returns:
            None: Because it does not use the state transition model.
        """
        return None


def initializer(config_path: str | Path):
    global user_func, workspace

    config = load_config(config_path)

    # Load the specified module from the specified python program.
    spec = importlib.util.spec_from_file_location("user_module", config.generic.python_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    user_func = getattr(module, config.generic.function)

    workspace = Path(config.generic.workspace).resolve()


def execute(args):
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
