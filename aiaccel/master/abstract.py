import multiprocessing

from aiaccel.module import AbstractModule
from aiaccel.master.evaluator.maximize import MaximizeEvaluator
from aiaccel.master.evaluator.minimize import MinimizeEvaluator
from aiaccel.master.verification.abstract import AbstractVerification
from aiaccel.util.logger import str_to_logging_level
from aiaccel.util.time_tools import get_time_now_object
from aiaccel.util.time_tools import get_time_string_from_object
import aiaccel
import logging
import time
from aiaccel.util.serialize import Serializer
from aiaccel.optimizer.create import create_optimizer
from aiaccel.scheduler.create import create_scheduler


class AbstractMaster(AbstractModule):
    """An abstract class for AbciMaster and LocalMaster.

    Attributes:
        loop_start_time (datetime.datetime): A stored loop starting time.
        optimizer_proc (subprocess.Popen): A reference for a subprocess of
            Optimizer.
        start_time (datetime.datetime): A stored starting time.
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
        self.options['process_name'] = 'master'

        super().__init__(self.options)
        self.alive_file = self.ws / aiaccel.dict_alive / aiaccel.alive_master
        self.logger = logging.getLogger('root.master')
        self.logger.setLevel(logging.DEBUG)
        self.exit_alive('master')

        self.set_logger(
            'root.master',
            self.dict_log / self.config.master_logfile.get(),
            str_to_logging_level(self.config.master_file_log_level.get()),
            str_to_logging_level(self.config.master_stream_log_level.get()),
            'Master   '
        )
        print(self.options)
        self.verification = AbstractVerification(self.options)
        self.sleep_time = self.config.sleep_time_master.get()
        self.goal = self.config.goal.get()
        self.trial_number = self.config.trial_number.get()
        self.serialize = Serializer(self.config, 'master', self.options)

        # optimizer
        self.o = create_optimizer(options['config'])(options)
        # scheduler
        self.s = create_scheduler(options['config'])(options)

        self.worker_o = multiprocessing.Process(target=self.o.start)
        self.worker_s = multiprocessing.Process(target=self.s.start)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None

        Raises:
            IndexError: Causes when expire the count which cannot confirm to
                run Optimizer and Scheduler.
        """
        super().pre_process()

        self.trial_id.initial(num=0)
        resume_trial_id = self.options['resume']
        if resume_trial_id is not None:
            self.storage.rollback_to_ready(resume_trial_id)
            self.storage.delete_trial_data_after_this(resume_trial_id)
            self.trial_id.initial(num=resume_trial_id)

        self.start_optimizer()
        self.start_scheduler()
        c = 0

        while (
            self.storage.alive.check_alive('optimizer') is False or
            self.storage.alive.check_alive('scheduler') is False
        ):
            time.sleep(1.0)
            c += 1

            if c >= self.config.init_fail_count.get():
                self.logger.error(f'Start process fails {self.config.init_fail_count.get()} times.')

                if self.storage.alive.check_alive('optimizer') is False:
                    raise IndexError('Could not start an optimizer process.')

                elif self.storage.alive.check_alive('scheduler') is False:
                    raise IndexError('Could not start an scheduler process.')

                else:
                    assert False

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
            evaluator = MaximizeEvaluator(self.options)
        elif self.goal.lower() == aiaccel.goal_minimize:
            evaluator = MinimizeEvaluator(self.options)
        else:
            self.logger.error('Invalid goal: {}.'.format(self.goal))
            raise ValueError('Invalid goal: {}.'.format(self.goal))

        evaluator.evaluate()
        evaluator.print()
        evaluator.save()

        # verification
        self.verification.verify()
        self.verification.save('final')
        self.storage.alive.set_any_process_state('master', 0)
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
                finishing_time = (now + (self.trial_number - hp_finished) * one_loop_time)
                end_estimated_time = get_time_string_from_object(finishing_time)
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
        """ Start the Optimizer process.

        Returns:
            None
        """
        self.worker_o.daemon = True
        self.worker_o.start()

    def start_scheduler(self) -> None:
        """ Start the Scheduler process.

        Returns:
            None
        """
        self.worker_s.daemon = True
        self.worker_s.start()

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
        self.get_each_state_count()

        if not self.storage.alive.check_alive('optimizer'):
            self.logger.info('Optimizer alive state is False')
            self.stop()
            return False

        if not self.storage.alive.check_alive('scheduler'):
            self.logger.info('Scheduler alive state is False')
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

        return True

    def other_process_is_alive(self) -> bool:
        if (
            not self.worker_o.is_alive() or
            not self.worker_s.is_alive()
        ):
            return False
        return True
