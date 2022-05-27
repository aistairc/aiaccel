from aiaccel.module import AbstractModule
from aiaccel.scheduler.algorithm import schdule_sampling
from aiaccel.scheduler.job.job_thread import Job
from aiaccel.util.filesystem import check_alive_file
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.filesystem import file_delete
from aiaccel.util.filesystem import get_basename
from aiaccel.util.filesystem import get_file_result
from aiaccel.util.filesystem import get_file_hp_finished
from aiaccel.util.filesystem import get_file_hp_ready
from aiaccel.util.filesystem import get_file_hp_running
from aiaccel.util.filesystem import interprocess_lock_file
from aiaccel.util.filesystem import load_yaml
from aiaccel.util.logger import str_to_logging_level
from pathlib import Path
from typing import Union
import aiaccel
import fasteners
import logging
import re
import time
from aiaccel.util.snapshot import SnapShot
from aiaccel.util.terminal import Terminal


def parse_hashname(command: str) -> str:
    """Parse a command string and extract an unique name.

    Args:
        command (str): A command string from ps command.

    Returns:
        str: An unique name.
    """
    args = re.split(' +', command)
    hash_key_index = args.index('--index')

    return args[hash_key_index + 1]


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
        super().__init__(self.options)

        self.config_path = Path(self.options['config']).resolve()
        self.set_logger(
            'root.scheduler',
            self.dict_log / self.config.scheduler_logfile.get(),
            str_to_logging_level(self.config.scheduler_file_log_level.get()),
            str_to_logging_level(self.config.scheduler_stream_log_level.get()),
            'Scheduler'
        )

        if self.options['dbg'] is True:
            self.config.silent_mode.set(False)
        else:
            self.remove_logger_handler()
            self.logfile = 'scheduler.log'
            self.set_logger(
                'root.scheduler',
                self.dict_log / self.logfile,
                logging.DEBUG,
                logging.CRITICAL,
                'Scheduler'
            )

        self.exit_alive(self.alive_scheduler)
        self.max_resource = self.config.num_node.get()
        self.available_resource = self.max_resource
        self.stats = []
        self.jobs = []
        self.algorithm = None
        self.sleep_time = self.config.sleep_time_scheduler.get()
        self.snapshot = SnapShot(self.ws, 'scheduler')
        self.job_status = {}
        self.job_status_filepath = self.ws / aiaccel.dict_log / "job_status.txt"

    def check_finished_hp(self) -> None:
        """Create finished hyper parameter files if result files can be found
            and running files are in running directory.

        Returns:
            None
        """
        with fasteners.InterProcessLock(
            interprocess_lock_file((self.ws / aiaccel.dict_hp), self.dict_lock)
        ):
            result_files = get_file_result(self.ws, self.dict_lock)
            result_names = [get_basename(f) for f in result_files]
            hp_running_files = get_file_hp_running(self.ws)

            for hp in hp_running_files:
                hp_base = get_basename(hp)
                if get_basename(hp) in result_names:
                    index = result_names.index(hp_base)
                    hp_content = load_yaml(hp, self.dict_lock)
                    self.logger.debug(
                        '{} is moved from running to finished'
                        .format(result_names[index])
                    )
                    result_content = load_yaml(
                        result_files[index],
                        self.dict_lock
                    )
                    hp_content['end_time'] = result_content['end_time']
                    hp_content['start_time'] = result_content['start_time']
                    hp_content['result'] = result_content['result']
                    if 'error' in result_content.keys():
                        hp_content['error'] = result_content['error']
                    # Cleate HpFinished File
                    create_yaml(
                        self.ws / aiaccel.dict_hp_finished / hp_base,
                        hp_content,
                        self.dict_lock
                    )

    def get_stats(self) -> None:
        """Updates the number of files in hp(hyper parameter) directories.

        Returns:
            None
        """
        self.get_dict_state()

    def start_job_thread(self, hp: Path) -> Union[Job, None]:
        """Start a new job thread.

        Args:
            hp (Path): A parameter file.

        Returns:
            Union[Job, None]: A reference for created job. It returns None if
                specified hyper parameter file already exists.
        """
        if not get_basename(hp) in [job['hashname'] for job in self.jobs]:
            th = Job(self.config, self.config_path, self, hp)
            self.jobs.append({'hashname': get_basename(hp), 'thread': th})
            self.logger.debug("Submit a job: {}".format(str(hp)))
            th.start()
            return th
        else:
            self.logger.error(
                'Specified hyperparemeter file already exists '
                'in job threads: {}'.format(hp)
            )
            return None

    def update_resource(self) -> None:
        """Update an available current resource number.

        Returns:
            None
        """
        succeed_threads = len([
            th for th in self.jobs
            if th['thread'].get_state_name() == 'Success'
        ])
        ready_threads = len(
            [
                th for th in self.jobs if th['thread'].get_state_name() in [
                    'Init', 'RunnerReady', 'RunnerChecking', 'RunnerConfirmed',
                    'RunnerFailed', 'RunnerFailure', 'Scheduling'
                ]
            ]
        )
        running_threads = len(self.jobs) - ready_threads - succeed_threads
        self.available_resource = max(0, self.max_resource - running_threads)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()
        self.algorithm = schdule_sampling.RamsomSampling(self.config)
        self.check_finished_hp()
        finished_hashnames =\
            [get_basename(f) for f in get_file_hp_finished(self.ws)]

        # Delete unknown result files
        with fasteners.InterProcessLock(
            interprocess_lock_file(self.ws / aiaccel.dict_hp, self.dict_lock)
        ):
            for f in get_file_result(self.ws, self.dict_lock):
                if not get_basename(f) in finished_hashnames:
                    file_delete(f)
                    self.logger.info(
                        'pre_process(): delete an unknown result file: {}'
                        .format(f)
                    )

            hp_running_files = get_file_hp_running(self.ws)

        for hp in hp_running_files:
            th = self.start_job_thread(hp)
            self.logger.info(
                'restart hp files in '
                'previous running directory: {}'
                .format(hp)
            )

            while th.get_state_name() != 'Scheduling':
                time.sleep(self.sleep_time)

            th.schedule()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        file_delete(self.ws / aiaccel.dict_alive / aiaccel.alive_scheduler, self.dict_lock)
        self.logger.info('Scheduler finished.')

    def loop_pre_process(self) -> None:
        """Called before entering a main loop process.

        Returns:
            None
        """
        while not self.check_work_directory():
            time.sleep(self.sleep_time)

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
        if not check_alive_file(self.alive_scheduler, self.dict_lock):
            self.logger.info('Alive file is deleted')
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
        hp_ready = get_file_hp_ready(self.ws, self.dict_lock)

        # Find a new hp
        for ready in hp_ready:
            if (
                not get_basename(ready) in
                    [job['hashname'] for job in self.jobs]
            ):
                self.start_job_thread(ready)

        # Select one hyper parameter
        scheduled_candidates = []

        for job in self.jobs:
            if job['thread'].get_state_name() == 'Scheduling':
                scheduled_candidates.append(job['thread'])

        self.logger.debug(
            'len(scheduled_candidates): {}'
            .format(len(scheduled_candidates))
        )
        self.logger.debug(
            'available_resource: {}'
            .format(self.available_resource)
        )

        selected_threads = self.algorithm.select_hp(
            scheduled_candidates,
            self.available_resource
        )

        self.logger.debug(
            'len(selected_threads): {}'
            .format(len(selected_threads))
        )

        if selected_threads is not None:
            for th in selected_threads:
                if th.get_state_name() == 'Scheduling':
                    th.schedule()
                    selected_threads.remove(th)

        for job in self.jobs:
            self.logger.info('name: {}, state: {}'.format(
                job['hashname'], job['thread'].get_state_name()))

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
        return True

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized scheduler objects.
        """
        dict_objects = {'loop_count': self.loop_count}
        if self.options['nosave'] is True:
            pass
        else:
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

        self.loop_count = loop_counts['scheduler']
        print(
            "({})set inner loop count: {}"
            .format('scheduler', self.loop_count)
        )

    def get_scheduler_status(self):
        for job in self.jobs:
            self.job_status[job['hashname']] = job['thread'].get_state_name()

    def check_error(self):
        self.get_scheduler_status()

        # Check state machin
        for hashname in self.job_status.keys():
            if "Failure" in self.job_status[hashname]:
                Terminal().print_error(
                    "Job: {} is Failed.({})\n"
                    "This is a fatal internal error. "
                    "Please review the configuration file. "
                    "In particular, we recommend that you "
                    "review the following items: "
                    "'{}', '{}', '{}', '{}'"
                    "'{}', '{}', '{}', '{}'"
                    .format(
                        hashname, self.job_status[hashname],
                        "cancel_timeout", "expire_timeout",
                        "finished_timeout", "job_timeout",
                        "kill_timeout", "batch_job_timeout",
                        "runner_timeout", "running_timeout"
                    )
                )
                # System quit
                return False

        # NOTE; There is room to reduce the unprocessability.
        create_yaml(
            self.job_status_filepath,
            self.job_status,
            dict_lock=None
        )
        return True
