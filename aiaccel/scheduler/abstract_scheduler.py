from __future__ import annotations

from pathlib import Path

from aiaccel.common import dict_result
from aiaccel.module import AbstractModule
from aiaccel.scheduler import RandomSampling
from aiaccel.scheduler import Job
from aiaccel.util import str_to_logging_level
from aiaccel.util import create_yaml


class AbstractScheduler(AbstractModule):
    """An abstract class for AbciScheduler and LocalScheduler.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
        config_path (Path): Path to the configuration file.
        algorithm (RandomSampling): A scheduling algorithm
            to select hyper parameters from a parameter pool.
        available_resource (int): An available current resource number.
        jobs (list[dict]): A list to store job dictionaries.
        max_resource (int): A max resource number.
        stats (list[dict]): A list of current status which is updated using ps
            command or qstat command.
    """

    def __init__(self, options: dict) -> None:
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

        self.max_resource = self.config.num_node.get()
        self.available_resource = self.max_resource
        self.stats = []
        self.jobs = []
        self.job_status = {}
        self.algorithm = None
        self.num_node = self.config.num_node.get()

    def change_state_finished_trials(self) -> None:
        """Create finished hyper parameter files if result files can be found
        and running files are in running directory.

        Returns:
            None
        """
        runnings = self.storage.trial.get_running()
        result_names = self.storage.result.get_result_trial_id_list()

        if result_names is None:
            return

        for running in runnings:
            if running in result_names:
                self.storage.trial.set_any_trial_state(trial_id=running, state='finished')

    def get_stats(self) -> None:
        """Updates the number of files in hp(hyper parameter) directories.

        Returns:
            None
        """
        self.get_each_state_count()

    def start_job(self, trial_id: int) -> Job | None:
        """Start a new job.

        Args:
            hp (Path): A parameter file path

        Returns:
            Job | None: A reference for created job. It returns None if
            specified hyper parameter file already exists.
        """
        trial_ids = [job.trial_id for job in self.jobs]
        if trial_id not in trial_ids:
            job = Job(self.config, self, self.create_model(), trial_id)
            self.jobs.append(job)
            self.logger.debug(f"Submit a job: {str(trial_id)}")
            job.main()
            return job
        else:
            self.logger.error(f'Specified trial {trial_id} is already running ')
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

        succeed_jobs = [
            job for job in self.jobs
            if job.get_state_name() == 'Success'
        ]
        ready_jobs = [
            job for job in self.jobs
            if job.get_state_name() in state_names
        ]
        num_running_jobs = len(self.jobs) - len(ready_jobs) - len(succeed_jobs)
        self.available_resource = max(0, self.max_resource - num_running_jobs)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        self.create_numpy_random_generator()
        self.resume()

        self.algorithm = RandomSampling(self.config)
        self.change_state_finished_trials()

        runnings = self.storage.trial.get_running()
        for running in runnings:
            job = self.start_job(running)
            self.logger.info(f'restart hp files in previous running directory: {running}')

            while job.get_state_name() != 'Scheduling':
                job.main()
            job.schedule()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        self.logger.info('Scheduler finished.')

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """

        if self.check_finished():
            return False

        self.get_stats()

        readies = self.storage.trial.get_ready()

        # find a new hp
        for ready in readies:
            if (ready not in [job.trial_id for job in self.jobs]):
                self.start_job(ready)

        scheduled_candidates = []
        for job in self.jobs:
            if job.get_state_name() == 'Scheduling':
                scheduled_candidates.append(job)

        selected_jobs = self.algorithm.select_hp(
            scheduled_candidates,
            self.available_resource,
            rng=self._rng
        )

        if len(selected_jobs) > 0:
            for job in selected_jobs:
                if job.get_state_name() == 'Scheduling':
                    self._serialize(job.trial_id)
                    job.schedule()
                    self.logger.debug(
                        f"trial id: {job.trial_id} has been scheduled."
                    )
                    selected_jobs.remove(job)

        for job in self.jobs:
            job.main()
            self.logger.info(f"name: {job.trial_id}, state: {job.get_state_name()}")

        self.get_stats()
        self.update_resource()
        self.print_dict_state()
        if self.all_done() is True:
            return False

        return True

    def parse_trial_id(self, command: str) -> str:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str: An unique name.
        """
        pass

    def check_error(self) -> bool:

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

    def all_done(self) -> bool:
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
        """ When in resume mode, load the previous optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if (
            self.options['resume'] is not None and
            self.options['resume'] > 0
        ):
            self._deserialize(self.options['resume'])

    def create_result_file(self, trial_id: int) -> None:
        """ Save the results in yaml format.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """

        file_hp_count_fmt = f'%0{self.config.name_length.get()}d'
        file_name = file_hp_count_fmt % trial_id + '.hp'

        content = self.storage.get_hp_dict(trial_id)
        result = self.storage.result.get_any_trial_objective(trial_id=trial_id)
        error = self.storage.error.get_any_trial_error(trial_id=trial_id)

        content['result'] = result

        if error is not None:
            content['error'] = error

        for i in range(len(content['parameters'])):
            content['parameters'][i]['value'] = content['parameters'][i]['value']

        result_file_path = self.ws / dict_result / file_name
        create_yaml(result_file_path, content)

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['jobs']
        return obj

    def create_model(self) -> AbstractModel | None:
        """Creates model object of state machine.

        Override with a Scheduler that uses a Model.
        For example, LocalScheduler, AbciScheduler, etc.
        By the way, PylocalScheduler does not use Model.

        Returns:
            LocalModel: LocalModel object.

            Should return None.
            For that purpose, it is necessary to modify TestAbstractScheduler etc significantly.
            So it returns LocalModel.

            # TODO: Fix TestAbstractScheduler etc to return None.
        """
        return LocalModel()
