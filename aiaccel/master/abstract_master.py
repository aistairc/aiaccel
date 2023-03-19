from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from aiaccel.config import is_multi_objective
from aiaccel.common import goal_maximize
from aiaccel.common import goal_minimize
from aiaccel.module import AbstractModule
from aiaccel.master import MaximizeEvaluator
from aiaccel.master import MinimizeEvaluator
from aiaccel.master import AbstractVerification
from aiaccel.util import str_to_logging_level
from aiaccel.util import get_time_now_object
from aiaccel.util import get_time_string_from_object


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
        verification (AbstractVerification): A verification object.
        logger (logging.Logger): Logger object.
        goal (str): Goal of optimization ('minimize' or 'maximize').
        trial_number (int): The number of trials.
        runner_files (list):
        stats (list):
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        self.start_time = get_time_now_object()
        self.loop_start_time: datetime | None = None
        self.options = options
        self.options['process_name'] = 'master'

        super().__init__(self.options)
        self.logger = logging.getLogger('root.master')
        self.logger.setLevel(logging.DEBUG)

        self.set_logger(
            'root.master',
            self.dict_log / self.config.master_logfile.get(),
            str_to_logging_level(self.config.master_file_log_level.get()),
            str_to_logging_level(self.config.master_stream_log_level.get()),
            'Master   '
        )

        self.verification = AbstractVerification(self.options)
        self.goal = self.config.goal.get()
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
        if not self.check_finished():
            return

        if is_multi_objective(self.config):
            self.logger.info('Multi-objective optimization does not run an evaluation process.')
        elif self.goal.lower() == goal_maximize:
            evaluator = MaximizeEvaluator(self.options)
        elif self.goal.lower() == goal_minimize:
            evaluator = MinimizeEvaluator(self.options)
        else:
            self.logger.error(f'Invalid goal: {self.goal}.')
            raise ValueError(f'Invalid goal: {self.goal}.')

        if is_multi_objective(self.config) is False:
            evaluator.evaluate()
            evaluator.print()
            evaluator.save()

        # verification
        self.verification.verify()
        self.verification.save('final')
        self.logger.info('Master finished.')

    def print_dict_state(self) -> None:
        """Display the number of yaml files in 'ready', 'running', and
        'finished' directries in hp directory.

        Returns:
            None
        """
        now = get_time_now_object()

        if self.loop_start_time is None:
            end_estimated_time = 'Unknown'
        else:
            looping_time = now - self.loop_start_time

            if self.hp_finished != 0:
                one_loop_time = (looping_time / self.hp_finished)
                hp_finished = self.hp_finished
                finishing_time = (now + (self.trial_number - hp_finished) * one_loop_time)
                end_estimated_time = get_time_string_from_object(finishing_time)
            else:
                end_estimated_time = 'Unknown'

        self.logger.info(
            f'{self.hp_finished}/{self.trial_number} finished, '
            f'ready: {self.hp_ready} ,'
            f'running: {self.hp_running}, '
            f'end estimated time: {end_estimated_time}'
        )

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.get_each_state_count()

        if self.hp_finished >= self.trial_number:
            return False

        self.get_stats()
        self.print_dict_state()
        # verification
        self.verification.verify()

        return True

    def get_stats(self) -> None:
        """Get a current status and update.

        Returns:
            None
        """
        return None

    def check_error(self) -> bool:
        """ Check to confirm if an error has occurred.

        Args:
            None

        Returns:
            bool: True if no error. False if with error.
        """
        return True
