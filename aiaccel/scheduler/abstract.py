from pathlib import Path
from typing import Union
import time
import aiaccel
from aiaccel.util.serialize import Serializer
from aiaccel.module import AbstractModule
from aiaccel.scheduler.algorithm import schdule_sampling
from aiaccel.scheduler.job.job_thread import Job
from aiaccel.evaluator.maximize import MaximizeEvaluator
from aiaccel.evaluator.minimize import MinimizeEvaluator
from aiaccel.verification.abstract import AbstractVerification
from aiaccel.util.time_tools import get_time_now_object
from aiaccel.util.time_tools import get_time_string_from_object
from aiaccel.util.logger import Logger


class AbstractScheduler(AbstractModule):
    """An abstract class for AbciScheduler and LocalScheduler.

    Attributes:
        algorithm (RandomSamplingSchedulingAlgorithm): A scheduling algorithm
            to select hyper parameters from a parameter pool.
        available_resource (int): An available current resource number.
        jobs (List[dict]): A list to store job dictionaries.
        job_lock (fastener.lock.ReadWriteLock): A lock object to manage
            exclusive control of job threads each other.
        max_resource (int): A max resource number.
        stats (List[dict]): A list of current status which is updated using ps
            command or qstat command.
    """

    def __init__(self, options: dict) -> None:
        """Initial method for AbstractScheduler.

        Args:
            config (str): A file name of a configuration.
        """
        self.options = options
        self.options['module_name'] = 'scheduler'
        super().__init__(self.options)

        self.config_path = Path(self.options['config']).resolve()

        self._logger = Logger(
            logger_name='root.scheduler',
            logfile_path=self.ws / aiaccel.dict_log / self.config.scheduler_logfile.get()
        )
        self.logger = self._logger.create_logger(
            file_level=self.config.scheduler_file_log_level.get(),
            stream_level=self.config.scheduler_stream_log_level.get(),
            module_type='Scheduler'
        )

        self.max_resource = self.config.num_node.get()
        self.trial_number = self.config.trial_number.get()
        self.available_resource = self.max_resource
        self.stats = []
        self.jobs = []
        self.job_status = {}
        self.algorithm = None
        self.serialize = Serializer(self.config, 'scheduler', self.options)
        self.verification = AbstractVerification(self.options)
        self.start_time = get_time_now_object()
        self.loop_start_time = None

    def change_state_finished_trials(self) -> None:
        """Create finished hyper parameter files if result files can be found
            and running files are in running directory.

        Returns:
            None
        """
        runnings = self.storage.trial.get_running()
        result_names = self.storage.result.get_result_trial_id_list()
        for running in runnings:
            if running in result_names:
                self.storage.trial.set_any_trial_state(trial_id=running, state='finished')

    def get_stats(self) -> None:
        """Updates the number of files in hp(hyper parameter) directories.

        Returns:
            None
        """
        self.get_each_state_count()

    def start_job_thread(self, trial_id: int) -> Union[Job, None]:
        """Start a new job thread.

        Args:
            hp (Path): A parameter file path

        Returns:
            Union[Job, None]: A reference for created job. It returns None if
                specified hyper parameter file already exists.
        """
        trial_ids = [job['trial_id'] for job in self.jobs]
        if trial_id not in trial_ids:
            th = Job(self.config, self.options, self, trial_id)
            self.jobs.append({'trial_id': trial_id, 'thread': th})
            self.logger.debug(f"Submit a job: {str(trial_id)}")
            th.start()
            return th
        else:
            self.logger.error(
                f'Specified hyperparemeter file already exists '
                f'in job threads: {trial_id}'
            )
            return None

    def update_resource(self) -> None:
        """Update an available current resource number.

        Returns:
            None
        """
        state_names = [
            'Init',
            'RunnerReady',
            'RunnerChecking',
            'RunnerConfirmed',
            'RunnerFailed',
            'RunnerFailure',
            'Scheduling'
        ]

        succeed_threads = [
            th for th in self.jobs
            if th['thread'].get_state_name() == 'Success'
        ]
        ready_threads = [
            th for th in self.jobs
            if th['thread'].get_state_name() in state_names
        ]
        num_running_threads = len(self.jobs) - len(ready_threads) - len(succeed_threads)
        self.available_resource = max(0, self.max_resource - num_running_threads)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        self.trial_id.initial(num=0)
        self.set_native_random_seed()
        self.set_numpy_random_seed()
        self.resume()

        self.algorithm = schdule_sampling.RamsomSampling(self.config)
        self.change_state_finished_trials()

        runnings = self.storage.trial.get_running()
        sleep_time = self.config.sleep_time.get()

        for running in runnings:
            th = self.start_job_thread(running)
            self.logger.info(f'restart hp files in previous running directory: {running}')

            while th.get_state_name() != 'Scheduling':
                time.sleep(sleep_time)

            th.schedule()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """

        if not self.check_finished():
            return

        goal = self.config.goal.get()
        if goal.lower() == aiaccel.goal_maximize:
            evaluator = MaximizeEvaluator(self.options)
        elif goal.lower() == aiaccel.goal_minimize:
            evaluator = MinimizeEvaluator(self.options)
        else:
            self.logger.error(f'Invalid goal: {goal}.')
            raise ValueError(f'Invalid goal: {goal}.')

        evaluator.evaluate()
        evaluator.print()
        evaluator.save()

        # verification
        self.verification.verify()
        self.verification.save('final')
        self.logger.info('Scheduler finished.')

    def loop_pre_process(self) -> None:
        """Called before entering a main loop process.

        Returns:
            None
        """
        self.loop_start_time = get_time_now_object()
        return None

    def loop_post_process(self) -> None:
        """Called after exiting a main loop process.

        Returns:
            None
        """
        if not self.check_finished():
            for job in self.jobs:
                job['thread'].stop()

            for job in self.jobs:
                job['thread'].join()

    def inner_loop_pre_process(self) -> bool:
        """Called before executing a main loop process. This process is
            repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        if self.check_finished():
            self.logger.info('All parameters have been done.')
            self.logger.info('Wait all threads finish...')
            for job in self.jobs:
                job['thread'].join()
            return False
        return True

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.get_stats()

        readies = self.storage.trial.get_ready()

        # find a new hp
        for ready in readies:
            if (ready not in [job['trial_id'] for job in self.jobs]):
                self.start_job_thread(ready)

        scheduled_candidates = []
        for job in self.jobs:
            if job['thread'].get_state_name() == 'Scheduling':
                scheduled_candidates.append(job['thread'])

        selected_threads = self.algorithm.select_hp(
            scheduled_candidates,
            self.available_resource
        )

        if len(selected_threads) > 0:
            for th in selected_threads:
                if th.get_state_name() == 'Scheduling':
                    self._serialize(th.trial_id)
                    th.schedule()
                    self.logger.debug(
                        f"trial id: {th.trial_id} has been scheduled."
                    )
                    selected_threads.remove(th)

        for i in reversed(range(len(self.jobs))):
            if self.jobs[i]['thread'].get_state_name().lower() == "success":
                del self.jobs[i]

        if (
            self.config.scheduler_file_log_level.get().lower() == "info" or
            self.config.scheduler_stream_log_level.get().lower() == "info"
        ):
            for job in self.jobs:
                self.logger.info(f"name: {job['trial_id']}, state: {job['thread'].get_state_name()}")

        return True

    def inner_loop_post_process(self) -> bool:
        """Called after exiting a main loop process. This process is repeated
            every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.get_stats()
        self.update_resource()
        self.print_dict_state()
        self.verification.verify()

        if self.all_done() is True:
            return False
        return True

    def _serialize(self, trial_id) -> None:

        self.serialize_datas = {'loop_count': self.loop_count}

        self.serialize.serialize(
            trial_id=trial_id,
            optimization_variables=self.serialize_datas,
            native_random_state=self.get_native_random_state(),
            numpy_random_state=self.get_numpy_random_state()
        )

    def _deserialize(self, trial_id: int) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        d = self.serialize.deserialize(trial_id)
        self.deserialize_datas = d['optimization_variables']

        if self.deserialize_datas['loop_count'] is None:
            return

        self.loop_count = self.deserialize_datas['loop_count']

        print(f"(scheduler)set inner loop count: {self.loop_count}")

    def parse_trial_id(self, command: str) -> str:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str: An unique name.
        """
        pass

    def check_error(self):

        # Check state machin
        jobstates = self.storage.jobstate.get_all_trial_jobstate()
        for trial_id in self.job_status.keys():
            for jobstate in jobstates:
                if (
                    jobstate['trial_id'] == trial_id and
                    "failure" in jobstate['jobstate'].lower()
                ):
                    self.logger.info(
                        f"Job: {trial_id} is Failed.({self.job_status[trial_id]})\n"
                        f"This is a fatal internal error. "
                        f"Please review the configuration file. "
                        f"In particular, we recommend that you "
                        f"review the following items: "
                        f"{'cancel_timeout'}, "
                        f"{'expire_timeout'}, "
                        f"{'finished_timeout'}, "
                        f"{'job_timeout'}, "
                        f"{'kill_timeout'}, "
                        f"{'batch_job_timeout'}, "
                        f"{'runner_timeout'}, "
                        f"{'running_timeout'}"
                    )
                    return False
        return True

    def all_done(self):
        done_states = [
            'Success',
            'HpCancelFailure',
            'KillFailure',
            'HpExpiredFailure',
            'HpFinishedFailure',
            'JobFailure',
            'HpRunningFailure',
            'RunnerFailure'
        ]

        jobstates = self.storage.jobstate.get_all_trial_jobstate()

        num_trials = 0
        for s in done_states:
            num_trials += jobstates.count(s)

        return (num_trials >= self.config.trial_number.get())

    def resume(self) -> None:
        """ When in resume mode, load the previous
                optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if (
            self.options['resume'] is not None and
            self.options['resume'] > 0
        ):
            self.storage.rollback_to_ready(self.options['resume'])
            self.storage.delete_trial_data_after_this(self.options['resume'])
            self.trial_id.initial(num=self.options['resume'])

            self._deserialize(self.options['resume'])

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
            f'{self.hp_finished}/{self.trial_number} finished, '
            f'ready: {self.hp_ready} ,'
            f'running: {self.hp_running}, '
            f'end estimated time: {end_estimated_time}'
        )
