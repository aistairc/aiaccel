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
        self.workspace = Path(self.config.workspace.get()).resolve()
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
            xs = self.run.get_any_trial_xs(trial_id)
            args.append([trial_id, xs])
            self._serialize(trial_id)
        for trial_id, xs, y, err, start_time, end_time in self.pool.imap_unordered(self.execute, args):
            self.run.report(trial_id, xs, y, err, start_time, end_time)
            self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')
            self.create_result_file(trial_id)

        return True

    def execute(self, args) -> None:
        trial_id, xs = args
        start_time = get_time_now()

        set_logging_file_for_trial_id(self.workspace, trial_id)
        y = None
        err = ""

        try:
            y = cast_y(user_func(xs), y_data_type=None)
        except BaseException as e:
            err = str(e)
            y = None
        else:
            err = ""
        finally:
            self.com.out(objective_y=y, objective_err=err)

        end_time = get_time_now()

        return trial_id, xs, y, err, start_time, end_time

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['run']
        del obj['pool']
        return obj


def initializer(config_path: str | Path):
    global user_func
    config = Config(config_path)

    # Loads the specified module from the specified python program.
    spec = importlib.util.spec_from_file_location("user_module", config.python_file.get())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    user_func = getattr(module, config.function.get())
