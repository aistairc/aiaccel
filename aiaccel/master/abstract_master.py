from aiaccel.module import AbstractModule
from aiaccel.master.evaluator.maximize_evaluator import MaximizeEvaluator
from aiaccel.master.evaluator.minimize_evaluator import MinimizeEvaluator
from aiaccel.master.verification.abstract_verification import \
    AbstractVerification
from aiaccel.util.filesystem import check_alive_file
from aiaccel.util.filesystem import file_delete
from aiaccel.util.logger import str_to_logging_level
from aiaccel.util.process import exec_runner
from aiaccel.util.process import OutputHandler
from aiaccel.util.time_tools import get_time_now_object
from aiaccel.util.time_tools import get_time_string_from_object
import aiaccel
import logging
import time
from aiaccel.util.snapshot import SnapShot


class AbstractMaster(AbstractModule):
    """An abstract class for AbciMaster and LocalMaster.

    Attributes:
        loop_start_time (datetime.datetime): A stored loop starting time.
        optimizer_proc (subprocess.Popen): A reference for a subprocess of
            Optimizer.
        start_time (datetime.datetime): A stored starting time.
        scheduler_proc (subprocess.Popen): A reference for a subprocess of
            Scheduler.
        verification (AbstractVerification): A verification object.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of AbstractMaster.

        Args:
            config (str): A file name of a configuration.
        """
        self.start_time = get_time_now_object()
        self.loop_start_time = None
        self.options = options

        super().__init__(self.options)

        self.alive_file = self.ws / aiaccel.dict_alive / aiaccel.alive_master
        self.exit_alive(self.alive_file)
        self.make_work_directory()
        self.set_logger(
            'root.master',
            self.dict_log / self.config.master_logfile.get(),
            str_to_logging_level(self.config.master_file_log_level.get()),
            str_to_logging_level(self.config.master_stream_log_level.get()),
            'Master   '
        )
        if self.options['dbg'] is True:
            self.config.silent_mode.set(False)
        else:
            self.remove_logger_handler()
            self.logfile = 'master.log'
            self.set_logger(
                'root.master',
                self.dict_log / self.logfile,
                logging.DEBUG,
                logging.CRITICAL,
                'Master    '
            )
        self.verification = AbstractVerification(self.config_path)
        self.optimizer_proc = None
        self.scheduler_proc = None
        self.sleep_time = self.config.sleep_time_master.get()
        self.goal = self.config.goal.get()
        self.trial_number = self.config.trial_number.get()
        self.snapshot = SnapShot(self.ws, 'master')

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None

        Raises:
            IndexError: Causes when expire the count which cannot confirm to
                run Optimizer and Scheduler.
        """
        super().pre_process()
        while not self.check_work_directory():
            time.sleep(self.sleep_time)

        self.start_optimizer()
        self.start_scheduler()
        c = 0

        # Wait till optimizer and scheduler start
        alive_optimizer = self.ws / aiaccel.dict_alive / aiaccel.alive_optimizer
        alive_scheduler = self.ws / aiaccel.dict_alive / aiaccel.alive_scheduler

        while (
            not check_alive_file(alive_optimizer, self.dict_lock) or
            not check_alive_file(alive_scheduler, self.dict_lock)
        ):
            time.sleep(self.sleep_time)
            c += 1

            if c >= self.config.init_fail_count.get():
                self.logger.error(
                    'Start process fails {} times.'
                    .format(self.config.init_fail_count.get())
                )
                raise IndexError(
                    'Could not start an optimizer or a scheduler process.'
                )

            if self.check_finished():
                break

            self.logger.debug('check alive loop')

    def post_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None

        Raises:
            ValueError: Causes when an invalid goal is set.
        """
        if not self.check_finished():
            return

        if self.goal.lower() == aiaccel.goal_maximize:
            evaluator = MaximizeEvaluator(self.config)
        elif self.goal.lower() == aiaccel.goal_minimize:
            evaluator = MinimizeEvaluator(self.config)
        else:
            self.logger.error('Invalid goal: {}.'.format(self.goal))
            raise ValueError('Invalid goal: {}.'.format(self.goal))

        evaluator.evaluate()
        evaluator.print()
        evaluator.save()

        # verification
        self.verification.verify()
        self.verification.save('final')
        file_delete(
            self.alive_file,
            self.dict_lock
        )
        self.logger.info('Master finished.')

    def print_dict_state(self):
        """ Display the number of yaml files in 'ready' 'running'
            and 'finished' directries in hp directory.

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
                finishing_time = (
                    now + (self.trial_number - hp_finished) * one_loop_time
                )
                end_estimated_time = get_time_string_from_object(
                    finishing_time
                )
            else:
                end_estimated_time = 'Unknown'

        self.logger.info(
            '{}/{} finished, ready: {}, running: {}, end estimated time: {}'
            .format(
                self.hp_finished,
                self.trial_number,
                self.hp_ready,
                self.hp_running,
                end_estimated_time
            )
        )

    def start_optimizer(self) -> None:
        """Start Optimizer.

        Returns:
            None
        """
        optimizer_command = self.config.optimizer_command.get().split(" ")
        optimizer_command.append('--config')
        optimizer_command.append(str(self.config_path))

        self.optimizer_proc = exec_runner(
            optimizer_command,
            self.config.silent_mode.get()
        )
        self.th_optimizer = OutputHandler(
            self,
            self.optimizer_proc,
            'Optimizer'
        )
        self.th_optimizer.start()

    def start_scheduler(self) -> None:
        """Start Scheduler

        Returns:
            None
        """
        scheduler_command = self.config.scheduler_command.get().split(" ")
        scheduler_command.append('--config')
        scheduler_command.append(str(self.config_path))

        self.scheduler_proc = exec_runner(
            scheduler_command,
            self.config.silent_mode.get()
        )
        self.th_scheduler = OutputHandler(
            self, self.scheduler_proc,
            'Scheduler'
        )
        self.th_scheduler.start()

    def loop_pre_process(self) -> None:
        """Called before entering a main loop process.

        Returns:
            None
        """
        self.loop_start_time = get_time_now_object()

    def loop_post_process(self) -> None:
        """Called after exiting a main loop process.

        Returns:
            None
        """
        while self.other_process_is_alive():
            self.logger.debug('Wait till optimizer or scheduler finished.')

        return

    def inner_loop_pre_process(self) -> bool:
        """Called before executing a main loop process. This process is
            repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.get_dict_state()
        if not check_alive_file(
            self.alive_file,
            self.dict_lock
        ):
            self.logger.info('The alive file of optimizer is deleted')
            return False

        if not self.other_process_is_alive():
            self.logger.info(
                'Optimizer or Schduler process is none'
            )
            self.stop()
            return False
        return True

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        if self.hp_finished >= self.trial_number:
            return False

        return True

    def inner_loop_post_process(self) -> bool:
        """Called after exiting a main loop process. This process is repeated
            every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.print_dict_state()
        # verification
        self.verification.verify()
        time.sleep(self.sleep_time)

        return True

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized master objects.
        """
        if self.options['nosave'] is True:
            pass
        else:
            dict_objects = {
                'start_time': self.start_time,
                'loop_start_time': self.loop_start_time
            }
            self.snapshot.save(
                self.curr_trial_number,
                self.loop_count,
                self.get_native_random_state(),
                self.get_numpy_random_state(),
                dict_objects
            )
            return dict_objects

    def _deserialize(self, dict_objects: dict) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """

        loop_counts = (
            self.snapshot.get_inner_loop_counter(self.options['resume'])
        )
        if loop_counts is None:
            return

        self.loop_count = loop_counts['master']
        print(
            "({})set inner loop count: {}"
            .format('master', self.loop_count)
        )

    def other_process_is_alive(self) -> bool:
        """ Check the optimizer and scheduler process are alive.

        Returns:
            bool
        """
        if(
            not self.optimizer_proc.poll() is None or
            not self.scheduler_proc.poll() is None
        ):
            return False
        return True
