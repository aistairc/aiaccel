from pathlib import Path
from typing import Union
import time
from aiaccel.util.terminal import Terminal
from aiaccel.util.serialize import Serializer
from aiaccel.module import AbstractModule
from aiaccel.scheduler.algorithm import schdule_sampling
from aiaccel.scheduler.job.job_thread import Job
from aiaccel.util.logger import str_to_logging_level


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
        self.options['process_name'] = 'scheduler'
        super().__init__(self.options)

        self.config_path = Path(self.options['config']).resolve()

        self.set_logger(
            'root.scheduler',
            self.dict_log / self.config.scheduler_logfile.get(),
            str_to_logging_level(self.config.scheduler_file_log_level.get()),
            str_to_logging_level(self.config.scheduler_stream_log_level.get()),
            'Scheduler'
        )

        self.exit_alive('scheduler')
        self.max_resource = self.config.num_node.get()
        self.available_resource = self.max_resource
        self.stats = []
        self.jobs = []
        self.algorithm = None
        self.sleep_time = self.config.sleep_time_scheduler.get()
        self.job_status = {}
        self.serialize = Serializer(self.config, 'scheduler', self.options)

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
        super().pre_process()
        self.algorithm = schdule_sampling.RamsomSampling(self.config)
        self.change_state_finished_trials()

        runnings = self.storage.trial.get_running()
        for running in runnings:
            th = self.start_job_thread(running)
            self.logger.info(f'restart hp files in previous running directory: {running}')

            while th.get_state_name() != 'Scheduling':
                time.sleep(self.sleep_time)

            th.schedule()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        self.storage.alive.set_any_process_state('scheduler', 0)
        self.logger.info('Scheduler finished.')

    def loop_pre_process(self) -> None:
        """Called before entering a main loop process.

        Returns:
            None
        """
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
        if not self.storage.alive.check_alive('scheduler'):
            self.logger.info('Scheduler alive state is False')
            return False

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
                    th.schedule()
                    self.logger.debug(
                        f"trial id: {th.trial_id} has been scheduled."
                    )
                    selected_threads.remove(th)

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
        if self.all_done() is True:
            return False
        return True

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized scheduler objects.
        """

        self.serialize_datas = {'loop_count': self.loop_count}

        if self.options['nosave'] is True:
            pass
        else:
            self.serialize.serialize(
                trial_id=self.current_max_trial_number,
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

    def update_scheduler_status(self):
        # states = [
        #     {
        #         "trial_id": job['trial_id'],
        #         "jobstate": job['thread'].get_state_name()
        #     } for job in self.jobs
        # ]
        # self.storage.jobstate.set_any_trial_jobstates(states=states)

        for job in self.jobs:
            self.storage.jobstate.set_any_trial_jobstate(
                trial_id=job['trial_id'],
                state=job['thread'].get_state_name()
            )

    def check_error(self):
        # self.update_scheduler_status()

        # Check state machin
        jobstates = self.storage.jobstate.get_all_trial_jobstate()
        for trial_id in self.job_status.keys():
            for jobstate in jobstates:
                if (
                    jobstate['trial_id'] == trial_id and
                    "failure" in jobstate['jobstate'].lower()
                ):
                    Terminal().print_error(
                        "Job: {} is Failed.({})\n"
                        "This is a fatal internal error. "
                        "Please review the configuration file. "
                        "In particular, we recommend that you "
                        "review the following items: "
                        "'{}', '{}', '{}', '{}'"
                        "'{}', '{}', '{}', '{}'"
                        .format(
                            trial_id, self.job_status[trial_id],
                            "cancel_timeout", "expire_timeout",
                            "finished_timeout", "job_timeout",
                            "kill_timeout", "batch_job_timeout",
                            "runner_timeout", "running_timeout"
                        )
                    )
                    return False

        # for trial_id in self.job_status.keys():
        #     if self.storage.jobstate.is_failure(int(trial_id)):
        #         Terminal().print_error(
        #             "Job: {} is Failed.({})\n"
        #             "This is a fatal internal error. "
        #             "Please review the configuration file. "
        #             "In particular, we recommend that you "
        #             "review the following items: "
        #             "'{}', '{}', '{}', '{}'"
        #             "'{}', '{}', '{}', '{}'"
        #             .format(
        #                 trial_id, self.job_status[trial_id],
        #                 "cancel_timeout", "expire_timeout",
        #                 "finished_timeout", "job_timeout",
        #                 "kill_timeout", "batch_job_timeout",
        #                 "runner_timeout", "running_timeout"
        #             )
        #         )
        #         return False
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
