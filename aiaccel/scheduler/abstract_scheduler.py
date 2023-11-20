from __future__ import annotations

from typing import Any

from omegaconf.dictconfig import DictConfig

from aiaccel.module import AbstractModule
from aiaccel.scheduler.job.job import Job
from aiaccel.scheduler.job.model.local_model import LocalModel
from aiaccel.util import Buffer, create_yaml, str_to_logging_level


class AbstractScheduler(AbstractModule):
    """An abstract class for AbciScheduler and LocalScheduler.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
        available_resource (int): An available current resource number.
        jobs (list[dict]): A list to store job dictionaries.
        max_resource (int): A max resource number.
        stats (list[dict]): A list of current status which is updated using ps
            command or qstat command.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, "scheduler")
        self.set_logger(
            "root.scheduler",
            self.workspace.log / self.config.logger.file.scheduler,
            str_to_logging_level(self.config.logger.log_level.scheduler),
            str_to_logging_level(self.config.logger.stream_level.scheduler),
            "Scheduler",
        )

        self.max_resource = self.config.resource.num_workers
        self.available_resource = self.max_resource
        self.stats: list[Any] = []
        self.jobs: list[Any] = []
        self.job_status: dict[Any, Any] = {}
        self.num_workers = self.config.resource.num_workers
        self.trial_number = self.config.optimize.trial_number
        self.start_trial_id = self.config.resume if self.config.resume is not None else 0
        self.buff = Buffer([trial_id for trial_id in range(self.start_trial_id, self.trial_number)])
        for trial_id in range(self.start_trial_id, self.trial_number):
            self.buff.d[trial_id].set_max_len(2)
        self.job_completed_count = 0

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
                self.storage.trial.set_any_trial_state(trial_id=running, state="finished")

    def get_stats(self) -> None:
        """Updates the number of files in hp(hyper parameter) directories.

        Returns:
            None
        """
        self.update_each_state_count()

    def start_job(self, trial_id: int) -> Any:
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
            self.logger.error(f"Specified trial {trial_id} is already running ")
            return None

    def update_resource(self) -> None:
        """Update an available current resource number.

        Returns:
            None
        """
        state_names = [
            "Init",
            "RunnerReady",
            "RunnerChecking",
            "RunnerConfirmed",
            "RunnerFailed",
            "RunnerFailure",
            "Scheduling",
        ]

        succeed_jobs = [job for job in self.jobs if job.get_state_name() == "Success"]
        ready_jobs = [job for job in self.jobs if job.get_state_name() in state_names]
        num_running_jobs = len(self.jobs) - len(ready_jobs) - len(succeed_jobs)
        self.available_resource = max(0, self.max_resource - num_running_jobs)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        self.write_random_seed_to_debug_log()
        self.resume()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        self.logger.info("Scheduler finished.")

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
            if ready not in [job.trial_id for job in self.jobs]:
                self.start_job(ready)
                self._serialize(ready)

        for job in self.jobs:
            job.main()
            state_name = job.get_state_name()
            if state_name in {"success", "failure", "timeout"}:
                self.job_completed_count += 1
                self.jobs.remove(job)
                continue
            # Only log if the state has changed.
            if job.trial_id in self.buff.d.keys():
                self.buff.d[job.trial_id].Add(state_name)
                if self.buff.d[job.trial_id].has_difference():
                    self.logger.info(f"name: {job.trial_id}, state: {state_name}")

        self.get_stats()
        self.update_resource()
        self.print_dict_state()

        if self.trial_number == self.job_completed_count:
            self.logger.info("All jobs are completed.")
            return False
        return True

    def parse_trial_id(self, command: str) -> str | None:
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
                if jobstate["trial_id"] == trial_id and "failure" in jobstate["jobstate"].lower():
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

    def resume(self) -> None:
        """When in resume mode, load the previous optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if self.config.resume is not None and self.config.resume > 0:
            self._deserialize(self.config.resume)

    def __getstate__(self) -> dict[str, Any]:
        obj = super().__getstate__()
        del obj["jobs"]
        return obj

    def create_model(self) -> Any:
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

    def evaluate(self) -> None:
        """Evaluate the result of optimization.

        Returns:
            None
        """

        best_trial_ids, _ = self.storage.get_best_trial(self.goals)
        if best_trial_ids is None:
            self.logger.error(f"Failed to output {self.workspace.best_result_file}.")
            return
        hp_results = []
        for best_trial_id in best_trial_ids:
            hp_results.append(self.storage.get_hp_dict(best_trial_id))

        create_yaml(self.workspace.best_result_file, hp_results, self.workspace.lock)

        finished = self.storage.get_num_finished()
        if self.config.optimize.trial_number >= finished:
            self.logger.info("Best hyperparameter is followings:")
            self.logger.info(hp_results)
