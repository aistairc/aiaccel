from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from aiaccel.common import file_final_result
from aiaccel.module import AbstractModule
from aiaccel.util import (create_yaml, get_time_now_object,
                          get_time_string_from_object, str_to_logging_level)


class AbstractMaster(AbstractModule):
    """An abstract class for AbciMaster and LocalMaster.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.
        loop_start_time (datetime.datetime): A stored loop starting time.
        optimizer_proc (subprocess.Popen): A reference for a subprocess of
            Optimizer.
        start_time (datetime.datetime): A stored starting time.
        logger (logging.Logger): Logger object.
        goals (list[str]): Goal of optimization ('minimize' or 'maximize').
        trial_number (int): The number of trials.
        runner_files (list):
        stats (list):
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        self.start_time = get_time_now_object()
        self.loop_start_time: datetime | None = None
        self.options = options
        self.options["process_name"] = "master"

        super().__init__(self.options)
        self.logger = logging.getLogger("root.master")
        self.logger.setLevel(logging.DEBUG)

        self.set_logger(
            'root.master',
            self.workspace.log / self.config.master_logfile.get(),
            str_to_logging_level(self.config.master_file_log_level.get()),
            str_to_logging_level(self.config.master_stream_log_level.get()),
            "Master   ",
        )

        if isinstance(self.config.goal.get(), str):
            self.goals = [self.config.goal.get()]
        else:
            self.goals = self.config.goal.get()

        self.trial_number = self.config.trial_number.get()

        self.runner_files: list[Path] | None = []
        self.stats: list[Any] = []

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None

        Raises:
            IndexError: Causes when expire the count which cannot confirm to
                run Optimizer and Scheduler.
        """
        self.loop_start_time = get_time_now_object()

        return

    def post_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None

        Raises:
            ValueError: Causes when an invalid goal is set.
        """

        self.evaluate()
        self.logger.info("Master finished.")

    def print_dict_state(self) -> None:
        """Display the number of yaml files in 'ready', 'running', and
        'finished' directries in hp directory.

        Returns:
            None
        """
        now = get_time_now_object()

        if self.loop_start_time is None:
            end_estimated_time = "Unknown"
        else:
            looping_time = now - self.loop_start_time

            if self.hp_finished != 0:
                one_loop_time = looping_time / self.hp_finished
                hp_finished = self.hp_finished
                finishing_time = now + (self.trial_number - hp_finished) * one_loop_time
                end_estimated_time = get_time_string_from_object(finishing_time)
            else:
                end_estimated_time = "Unknown"

        self.logger.info(
            f"{self.hp_finished}/{self.trial_number} finished, "
            f"ready: {self.hp_ready} ,"
            f"running: {self.hp_running}, "
            f"end estimated time: {end_estimated_time}"
        )

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.update_each_state_count()

        if self.hp_finished >= self.trial_number:
            return False

        self.get_stats()
        self.print_dict_state()

        return True

    def get_stats(self) -> None:
        """Get a current status and update.

        Returns:
            None
        """
        return None

    def check_error(self) -> bool:
        """Check to confirm if an error has occurred.

        Args:
            None

        Returns:
            bool: True if no error. False if with error.
        """
        return True

    def evaluate(self) -> None:
        """Evaluate the result of optimization.

        Returns:
            None
        """

        best_trial_ids, _ = self.storage.get_best_trial(self.goals)
        if best_trial_ids is None:
            raise ValueError("No best trial found.")

        hp_results = []

        for best_trial_id in best_trial_ids:
            hp_results.append(self.storage.get_hp_dict(best_trial_id))

        self.logger.info('Best hyperparameter is followings:')
        self.logger.info(hp_results)

        path = self.workspace.result / file_final_result
        create_yaml(path, hp_results, self.workspace.lock)
