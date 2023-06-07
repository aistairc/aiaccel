from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf.dictconfig import DictConfig

from aiaccel.common import datetime_format
from aiaccel.module import AbstractModule
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.logger import str_to_logging_level


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

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, "master")
        self.start_time = datetime.now()
        self.loop_start_time: datetime | None = None
        self.logger = logging.getLogger("root.master")
        self.logger.setLevel(logging.DEBUG)
        self.set_logger(
            "root.master",
            self.workspace.log / self.config.logger.file.master,
            str_to_logging_level(self.config.logger.log_level.master),
            str_to_logging_level(self.config.logger.stream_level.master),
            "Master   ",
        )

        self.goals = [item.value for item in self.config.optimize.goal]
        self.trial_number: int = self.config.optimize.trial_number

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
        self.loop_start_time = datetime.now()

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
        now = datetime.now()

        if self.loop_start_time is None:
            end_estimated_time = "Unknown"
        else:
            looping_time = now - self.loop_start_time

            if self.hp_finished != 0:
                one_loop_time = looping_time / self.hp_finished
                hp_finished = self.hp_finished
                finishing_time = now + (self.trial_number - hp_finished) * one_loop_time
                end_estimated_time = finishing_time.strftime(datetime_format)
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
            self.logger.error(f"Failed to output {self.workspace.final_result_file}.")
            return

        hp_results = []

        for best_trial_id in best_trial_ids:
            hp_results.append(self.storage.get_hp_dict(best_trial_id))

        self.logger.info("Best hyperparameter is followings:")
        self.logger.info(hp_results)

        create_yaml(self.workspace.final_result_file, hp_results, self.workspace.lock)
