from __future__ import annotations
from aiaccel.abci.batch import create_abci_batch_file
from aiaccel.abci.qsub import create_qsub_command
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.filesystem import interprocess_lock_file
from aiaccel.util.retry import retry
from aiaccel.util.process import exec_runner
from aiaccel.util.process import kill_process
from aiaccel.util.process import OutputHandler
from aiaccel.util.time_tools import get_time_now_object
from aiaccel.util.time_tools import get_time_delta
from aiaccel.util.trialid import TrialId
from aiaccel.wrapper_tools import create_runner_command
from aiaccel.util.wd import get_cmd_array  # wd/
from aiaccel.util.buffer import Buffer
from enum import Enum
from pathlib import Path
from transitions import Machine
from transitions.extensions.states import add_state_features, Tags
from typing import Union, TYPE_CHECKING
import aiaccel
import fasteners
import logging
import threading
import time
if TYPE_CHECKING:
    from aiaccel.scheduler.abci import AbciScheduler
    from aiaccel.scheduler.local import LocalScheduler
from aiaccel.config import Config
from aiaccel.storage.storage import Storage


JOB_STATES = [
    {'name': 'Init'},
    {'name': 'RunnerReady'},
    {'name': 'RunnerChecking'},
    {'name': 'RunnerConfirmed'},
    {'name': 'RunnerFailed'},
    {'name': 'RunnerFailure'},
    {'name': 'Scheduling'},
    {'name': 'HpRunningReady'},
    {'name': 'HpRunningChecking'},
    {'name': 'HpRunningConfirmed'},
    {'name': 'HpRunningFailed'},
    {'name': 'HpRunningFailure'},
    {'name': 'JobReady'},
    {'name': 'JobChecking'},
    {'name': 'JobFailed'},
    {'name': 'JobConfirmed'},
    {'name': 'JobFailure'},
    {'name': 'WaitResult'},
    {'name': 'Result'},
    {'name': 'HpFinishedReady'},
    {'name': 'HpFinishedChecking'},
    {'name': 'HpFinishedFailed'},
    {'name': 'HpFinishedConfirmed'},
    {'name': 'HpFinishedFailure'},
    {'name': 'HpExpireReady'},
    {'name': 'HpExpireChecking'},
    {'name': 'HpExpireConfirmed'},
    {'name': 'HpExpireFailed'},
    {'name': 'HpExpireFailure'},
    {'name': 'HpExpiredFailure'},
    {'name': 'Canceling'},
    {'name': 'KillReady'},
    {'name': 'KillChecking'},
    {'name': 'KillConfirmed'},
    {'name': 'KillFailed'},
    {'name': 'KillFailure'},
    {'name': 'CheckResult'},
    {'name': 'HpCancelReady'},
    {'name': 'HpCancelChecking'},
    {'name': 'HpCancelConfirmed'},
    {'name': 'HpCancelFailed'},
    {'name': 'HpCancelFailure'},
    {'name': 'Canceled'},
    {'name': 'Success'},
]

JOB_TRANSITIONS = [
    {
        'trigger': 'next',
        'source': 'Init',
        'dest': 'RunnerReady',
        'after': 'after_runner'
    },
    {
        'trigger': 'next',
        'source': 'RunnerReady',
        'dest': 'RunnerChecking',
        'before': 'before_runner_create'
    },
    {
        'trigger': 'next',
        'source': 'RunnerChecking',
        'dest': 'RunnerConfirmed',
        'conditions': 'conditions_runner_confirmed'
    },
    {
        'trigger': 'expire',
        'source': 'RunnerChecking',
        'dest': 'RunnerFailed',
        'before': 'before_failed'
    },
    {
        'trigger': 'next',
        'source': 'RunnerFailed',
        'dest': 'RunnerReady'
    },
    {
        'trigger': 'expire',
        'source': 'RunnerFailed',
        'dest': 'RunnerFailure'
    },
    {
        'trigger': 'next',
        'source': 'RunnerConfirmed',
        'dest': 'Scheduling',
        'after': 'after_confirmed'
    },
    {
        'trigger': 'schedule',
        'source': 'Scheduling',
        'dest': 'HpRunningReady',
        'after': 'after_running'
    },
    {
        'trigger': 'next',
        'source': 'HpRunningReady',
        'dest': 'HpRunningChecking',
        'before': 'change_state'
    },
    {
        'trigger': 'next',
        'source': 'HpRunningChecking',
        'dest': 'HpRunningConfirmed',
        'conditions': 'conditions_confirmed'
    },
    {
        'trigger': 'expire',
        'source': 'HpRunningChecking',
        'dest': 'HpRunningFailed',
        'before': 'before_failed'
    },
    {
        'trigger': 'next',
        'source': 'HpRunningFailed',
        'dest': 'HpRunningReady'
    },
    {
        'trigger': 'expire',
        'source': 'HpRunningFailed',
        'dest': 'HpRunningFailure'
    },
    {
        'trigger': 'next',
        'source': 'HpRunningConfirmed',
        'dest': 'JobReady',
        'after': 'after_job'
    },
    {
        'trigger': 'next',
        'source': 'JobReady',
        'dest': 'JobChecking',
        'before': 'before_job_submitted'
    },
    {
        'trigger': 'next',
        'source': 'JobChecking',
        'dest': 'JobConfirmed',
        'conditions': 'conditions_job_confirmed'
    },
    {
        'trigger': 'expire',
        'source': 'JobChecking',
        'dest': 'JobFailed',
        'before': 'before_failed'
    },
    {
        'trigger': 'next',
        'source': 'JobFailed',
        'dest': 'JobReady'
    },
    {
        'trigger': 'expire',
        'source': 'JobFailed',
        'dest': 'JobFailure'
    },
    {
        'trigger': 'next',
        'source': 'JobConfirmed',
        'dest': 'WaitResult',
        'after': 'after_wait_result'
    },
    {
        'trigger': 'next',
        'source': 'WaitResult',
        'dest': 'Result',
        'conditions': 'conditions_result',
        'after': 'after_result'
    },
    {
        'trigger': 'expire',
        'source': 'WaitResult',
        'dest': 'HpExpireReady',
        'after': 'after_expire'
    },
    {
        'trigger': 'next',
        'source': 'Result',
        'dest': 'HpFinishedReady',
        'after': 'after_finished'
    },
    {
        'trigger': 'next',
        'source': 'HpFinishedReady',
        'dest': 'HpFinishedChecking',
        'before': 'before_finished'
    },
    {
        'trigger': 'next',
        'source': 'HpFinishedChecking',
        'dest': 'HpFinishedConfirmed',
        'conditions': 'conditions_confirmed'
    },
    {
        'trigger': 'expire',
        'source': 'HpFinishedChecking',
        'dest': 'HpFinishedFailed',
        'before': 'before_failed'
    },
    {
        'trigger': 'next',
        'source': 'HpFinishedFailed',
        'dest': 'HpFinishedReady'
    },
    {
        'trigger': 'expire',
        'source': 'HpFinishedFailed',
        'dest': 'HpFinishedFailure'
    },
    {
        'trigger': 'next',
        'source': 'HpFinishedConfirmed',
        'dest': 'Success',
        'after': 'after_confirmed'
    },
    {
        'trigger': 'next',
        'source': 'HpExpireReady',
        'dest': 'HpExpireChecking',
        'before': 'change_state'
    },
    {
        'trigger': 'next',
        'source': 'HpExpireChecking',
        'dest': 'HpExpireConfirmed',
        'conditions': 'conditions_confirmed'
    },
    {
        'trigger': 'expire',
        'source': 'HpExpireChecking',
        'dest': 'HpExpireFailed',
        'before': 'before_failed'
    },
    {
        'trigger': 'next',
        'source': 'HpExpireFailed',
        'dest': 'HpExpireReady'
    },
    {
        'trigger': 'expire',
        'source': 'HpExpireFailed',
        'dest': 'HpExpireFailure'
    },
    {
        'trigger': 'next',
        'source': 'HpExpireConfirmed',
        'dest': 'Scheduling',
        'after': 'after_confirmed'
    },
    {
        'trigger': 'next',
        'source': 'Canceling',
        'dest': 'KillReady',
        'after': 'after_kill'
    },
    {
        'trigger': 'next',
        'source': 'KillReady',
        'dest': 'KillChecking',
        'before': 'before_kill_submitted'
    },
    {
        'trigger': 'next',
        'source': 'KillChecking',
        'dest': 'KillConfirmed',
        'conditions': 'conditions_kill_confirmed'
    },
    {
        'trigger': 'expire',
        'source': 'KillChecking',
        'dest': 'KillFailed',
        'before': 'before_failed'
    },
    {
        'trigger': 'next',
        'source': 'KillFailed',
        'dest': 'KillReady'
    },
    {
        'trigger': 'expire',
        'source': 'KillFailed',
        'dest': 'KillFailure'
    },
    {
        'trigger': 'next',
        'source': 'KillConfirmed',
        'dest': 'CheckResult',
        'after': 'after_check_result'
    },
    {
        'trigger': 'next',
        'source': 'CheckResult',
        'dest': 'Result',
        'conditions': 'conditions_result'
    },
    {
        'trigger': 'expire',
        'source': 'CheckResult',
        'dest': 'HpCancelReady',
        'after': 'after_cancel'
    },
    {
        'trigger': 'next',
        'source': 'HpCancelReady',
        'dest': 'HpCancelChecking',
        'prepare': 'prepare_expire',
        'before': 'change_state'
    },
    {
        'trigger': 'next',
        'source': 'HpCancelChecking',
        'dest': 'HpCancelConfirmed',
        'conditions': 'conditions_confirmed'
    },
    {
        'trigger': 'expire',
        'source': 'HpCancelChecking',
        'dest': 'HpCancelFailed',
        'before': 'before_failed'
    },
    {
        'trigger': 'next',
        'source': 'HpCancelFailed',
        'dest': 'HpCancelReady'
    },
    {
        'trigger': 'expire',
        'source': 'HpCancelFailed',
        'dest': 'HpCancelFailure'
    },
    {
        'trigger': 'next',
        'source': 'HpCancelConfirmed',
        'dest': 'Canceled',
        'after': 'after_confirmed'
    },
    {
        'trigger': 'cancel',
        'source': [
            'Init',
            'RunnerReady',
            'RunnerChecking',
            'RunnerConfirmed',
            'RunnerFailed',
            'HpRunningReady',
            'HpRunningChecking',
            'HpRunningConfirmed',
            'HpRunningFailed',
            'JobReady',
            'JobChecking',
            'JobConfirmed',
            'JobFailed',
            'HpFinishedReady',
            'HpFinishedChecking',
            'HpFinishedConfirmed',
            'HpFinishedFailed',
            'HpExpireReady',
            'HpExpireChecking',
            'HpExpireConfirmed',
            'HpExpireFailed',
            'Scheduling',
            'WaitResult',
            'Result'
        ],
        'dest': 'Canceling'
    }
]

job_lock = threading.Lock()


@add_state_features(Tags)
class CustomMachine(Machine):
    pass


class Model(object):

    # Common
    def after_confirmed(self, obj: 'Job') -> None:
        """State transition of 'after_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        obj.from_file = None
        obj.to_file = None
        obj.threshold_timeout = None
        obj.threshold_retry = None
        obj.count_retry = 0

    def before_failed(self, obj: 'Job') -> None:
        """State transition of 'before_failed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        obj.count_retry += 1

    def conditions_confirmed(self, obj) -> bool:
        """State transition of 'conditions_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target file exists or not.
        """
        with obj.lock:
            any_trial_state = obj.storage.trial.get_any_trial_state(trial_id=obj.trial_id)
        return (obj.next_state == any_trial_state)

    def change_trial_state(self, obj: 'Job') -> None:
        """State transition of 'change_trial_state'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        with obj.lock:
            obj.storage.trial.set_any_trial_state(
                trial_id=obj.trial_id,
                state=obj.next_trial_state
            )
        return

    def get_runner_file(self, obj: 'Job') -> None:
        return obj.ws / aiaccel.dict_runner / 'run_{}.sh'.format(obj.trial_id_str)

    # Runner
    def after_runner(self, obj: 'Job') -> None:
        """State transition of 'after_runner'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        obj.to_file = self.get_runner_file(obj)
        obj.next_state = 'running'
        obj.threshold_timeout = (
            get_time_now_object() +
            get_time_delta(obj.runner_timeout)
        )
        obj.threshold_retry = obj.runner_retry

    def before_runner_create(self, obj: 'Job') -> None:
        """State transition of 'before_runner_create'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        if obj.is_local():
            return

        commands = create_runner_command(
            obj.config.job_command.get(),
            obj.content,
            str(obj.trial_id),
            obj.config_path,
            obj.options
        )

        with obj.lock:
            create_abci_batch_file(
                obj.to_file,
                obj.config.job_script_preamble.get(),
                commands,
                obj.dict_lock
            )

    def conditions_runner_confirmed(self, obj: 'Job') -> bool:
        """State transition of 'conditions_runner_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target file exists or not. But always true if local
                execution.
        """
        if obj.is_local():
            return True

        @retry(_MAX_NUM=60, _DELAY=1.0)
        def _conditions_runner_confirmed(_obj):
            with _obj.lock:
                lockpath = interprocess_lock_file(_obj.to_file, _obj.dict_lock)
                with fasteners.InterProcessLock(lockpath):
                    return _obj.to_file.exists()

        return _conditions_runner_confirmed(obj)

    # HpRunning
    def after_running(self, obj: 'Job') -> None:
        """State transition of 'after_running'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        obj.next_state = "running"
        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.running_timeout)
        )
        obj.threshold_retry = obj.running_retry

    # JobRun
    def after_job(self, obj: 'Job') -> None:
        """State transition of 'after_job'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        self.after_confirmed(obj)
        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.job_timeout)
        )
        obj.threshold_retry = obj.job_retry

    def before_job_submitted(self, obj: 'Job') -> None:
        """State transition of 'before_job_submitted'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        if obj.is_local():
            runner_command = create_runner_command(
                obj.config.job_command.get(),
                obj.content,
                str(obj.trial_id),
                obj.config_path,
                obj.options
            )
            # wd/
            wd_cmd = get_cmd_array()
            if wd_cmd is not None:
                runner_command[0:0] = wd_cmd
        else:
            runner_file = self.get_runner_file(obj)
            runner_command = create_qsub_command(
                obj.config,
                str(runner_file)
            )

        obj.proc = exec_runner(
            runner_command,
            bool(obj.config.silent_mode.get())
        )

        obj.th_oh = OutputHandler(
            obj.scheduler,
            obj.proc,
            'Job'
        )
        obj.th_oh.start()

    def conditions_job_confirmed(self, obj: 'Job') -> bool:
        """State transition of 'conditions_job_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target job is finished or not.
        """
        for state in obj.scheduler.stats:
            # state: {
            #     'job-ID': '2481',
            #     'prior': None,
            #     'user': '3 member',
            #     'state': 'R+   Wed May 25 13:5',
            #     'queue': None,
            #     'jclass': None,
            #     'slots': None,
            #     'ja-task-ID': None,
            #     'name': '2 python user.py --trial_id 0 --config config.yaml --x1=1.0 --x2=1.0',
            #     'submit/start at': '4:11 202'
            # }
            if obj.trial_id == int(obj.scheduler.parse_trial_id(state['name'])):
                return True
        else:
            # confirm whether the result file exists or not (this means the job finished quickly
            with obj.lock:
                trial_ids = obj.storage.result.get_result_trial_id_list()
            return obj.trial_id in trial_ids

    # Result
    def after_result(self, obj: 'Job') -> None:
        """State transition of 'after_result'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        self.after_confirmed(obj)

    def after_wait_result(self, obj: 'Job') -> None:
        """State transition of 'after_wait_result'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        self.after_confirmed(obj)
        obj.threshold_timeout = (get_time_now_object() + get_time_delta(obj.batch_job_timeout))
        obj.threshold_retry = obj.result_retry

    def conditions_result(self, obj: 'Job') -> bool:
        """State transition of 'conditions_result'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target is in result files or not.
        """
        with obj.lock:
            objective = obj.storage.result.get_any_trial_objective(trial_id=obj.trial_id)
        return (objective is not None)

    # Finished
    def after_finished(self, obj: 'Job') -> None:
        """State transition of 'after_finished'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        self.after_confirmed(obj)
        # obj.from_file = obj.ws / aiaccel.dict_hp_running / obj.hp_file.name
        # obj.to_file = obj.ws / aiaccel.dict_hp_finished / obj.hp_file.name
        obj.next_state = 'finished'
        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.finished_timeout)
        )
        obj.threshold_retry = obj.finished_retry

    def before_finished(self, obj: 'Job') -> None:
        """State transition of 'before_finished'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """

        self.change_state(obj)

        with obj.lock:
            result = obj.storage.result.get_any_trial_objective(trial_id=obj.trial_id)
            error = obj.storage.error.get_any_trial_error(trial_id=obj.trial_id)
            content = obj.storage.get_hp_dict(trial_id_str=obj.trial_id_str)
            content['result'] = result

            if error is not None:
                content['error'] = error

            create_yaml(obj.result_file_path, content)

    # Expire
    def after_expire(self, obj: 'Job') -> None:
        """State transition of 'after_expire'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """

        obj.next_state = "ready"
        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.expire_timeout)
        )
        obj.threshold_retry = obj.expire_retry

    # Kill
    def after_kill(self, obj: 'Job') -> None:
        """State transition of 'after_kill'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.kill_timeout)
        )
        obj.threshold_retry = obj.kill_retry

    def before_kill_submitted(self, obj: 'Job') -> None:
        """State transition of 'before_kill_submitted'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): Ab object.

        Returns:
            None
        """
        for state in obj.scheduler.stats:
            # if obj.trial_id == state['name']:
            if obj.trial_id == int(obj.scheduler.parse_trial_id(state['name'])):
                kill_process(state['job-ID'])
        else:
            logger = logging.getLogger('root.scheduler.job')
            logger.warning('Not matched job trial_id: {}'.format(obj.trial_id))

    def conditions_kill_confirmed(self, obj: 'Job') -> bool:
        """State transition of 'conditions_kill_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target is killed or not.
        """
        for state in obj.scheduler.stats:
            # if obj.trial_id == state['name']:
            if obj.trial_id == int(obj.scheduler.parse_trial_id(state['name'])):
                return False
        else:
            return True

    # Check result
    def after_check_result(self, obj: 'Job') -> None:
        """State transition of 'after_check_result'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """
        self.after_confirmed(obj)
        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.batch_job_timeout)
        )
        obj.threshold_retry = obj.result_retry

    # Cancel
    def after_cancel(self, obj: 'Job') -> None:
        """State transition of 'after_cancel'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            None
        """

        # @retry(_MAX_NUM=60, _DELAY=1.0)
        # def _get_hp_ready_files(_obj):
        #     with _obj.lock.read_lock():
        #         ready_files = get_file_hp_ready(_obj.ws, _obj.dict_lock)
        #     return ready_files

        # @retry(_MAX_NUM=60, _DELAY=1.0)
        # def _get_hp_running_files(_obj):
        #     with _obj.lock.read_lock():
        #         running_files = get_file_hp_running(_obj.ws, _obj.dict_lock)
        #     return running_files

        # self.after_confirmed(obj)
        # # Search hp file in ready or running directory
        # ready_files = _get_hp_ready_files(obj)
        # running_files = _get_hp_running_files(obj)

        # if obj.trial_id in [get_basename(f) for f in ready_files]:
        #     obj.from_file = obj.ws / aiaccel.dict_hp_ready / obj.hp_file.name

        # elif obj.trial_id in [get_basename(f) for f in running_files]:
        #     obj.from_file = obj.ws / aiaccel.dict_hp_running / obj.hp_file.name

        # else:
        #     logger = logging.getLogger('root.scheduler.job')
        #     logger.warning(
        #         'Could not find any files trial_id: {}'
        #         .format(obj.trial_id)
        #     )

        # obj.to_file = obj.ws / aiaccel.dict_hp_ready / obj.hp_file.name
        # obj.threshold_timeout = (
        #     get_time_now_object() + get_time_delta(obj.expire_timeout)
        # )
        # obj.threshold_retry = obj.expire_retry
        with obj.lock:
            if (
                obj.storage.is_ready(obj.trial_id) or
                obj.storage.is_running(obj.trial_id)
            ):
                obj.storage.trial.set_any_trial_state(trial_id=obj.trial_id, state='ready')
            else:
                logger = logging.getLogger('root.scheduler.job')
                logger.warning('Could not find any trial_id: {}'.format(obj.trial_id))

        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.expire_timeout)
        )
        obj.threshold_retry = obj.expire_retry

    def change_state(self, obj: 'Job'):
        with obj.lock:
            obj.storage.trial.set_any_trial_state(trial_id=obj.trial_id, state=obj.next_state)


class Job(threading.Thread):
    """A job thread to manage running jobs on local computer or ABCI.

    ToDo: Confirm the state transition especially timeout expire and retry
        expire. Retry expire works well?

    Attributes:
        - config (ConfileWrapper): A configuration object.

        - ws (Path): A path of a workspace.

        - dict_lock (Path): A directory to store lock files.

        - runner_timeout (int):
            Timeout seconds to transit the state
            from RunnerChecking to RunnerFailed.

        - running_timeout (int):
            Timeout seconds to transit the state
            from HpRunningChecking to HpRunningFailed.

        - job_timeout (int):
            Timeout seconds to transit the state
            from JobChecking to JobFailed.

        - batch_job_timeout (int):
            Timeout seconds to transit the state
            from WaitResult to HpExpireReady.

        - finished_timeout (int):
            Timeout seconds to transit the state
            from HpFinishedChecking to HpFinishedFailed.

        - kill_timeout (int):
            Timeout seconds to transit the state
            from KillChecking to KillFailed.

        - cancel_timeout (int):
            Timeout seconds to transit the state
            from HpCancelChecking to HpCancelFailed.

        - expire_timeout (int):
            Timeout seconds to transit the state
            from HpExpireChecking to HpExpireFailed.

        - runner_retry (int):
            Max retry counts to transit the state
            from RunnerFailed to RunnerFailure.

        - running_retry (int):
            Max retry counts to transit the state
            from HpRunningFailed to HpRunningFailure.

        - job_retry (int):
            Max retry counts to transit the state
            from JobFailed to JobFailure.

        - result_retry (int):
            Max retry counts to transit the state
            from RunnerFailed to RunnerFailure.

        - finished_retry (int):
            Max retry counts to transit the state
            from HpFinishedFailed to HpFinishedFailure.

        - kill_retry (int):
            Max retry counts to transit the state
            from KillFailed to KillFailure.

        - cancel_retry (int):
            Max retry counts to transit the state
            from HpCancelFailed to HpCancelFailure.

        - expire_retry (int):
            Max retry counts to transit the state
            from HpExpireFailed to HpExpireFailure.

        - threshold_timeout (int):
            A timeout threshold for each state.
            A new value is stored each state transition if necessary.

        - threshold_retry (int):
            A retry threshold for each state.
            A new value is stored each state transition if necessary.

        - count_retry (int):
            A current retry count. This is compared with threshold_retry.

        - job_loop_duration (int): A sleep time each job loop.

        - model (Model): A model object of state transitions.

        - machine (CustomMachine): A state machine object.

        - c (int): A loop counter.

        - scheduler (Union[LocalScheduler, AbciScheduler]):
            A reference for scheduler object.

        - lock (fastener.lock.ReadWriteLock):
            A reference for scheduler lock object.
            This is for exclusive control of job threads each other.

        - hp_file (Path): A hyper parameter file for this job.

        - trial_id (str): A unique name of this job.

        - from_file (Path):
            A temporal file path to be used for each state transition.
            For example, it's used for file moving.

        - to_file (Path):
            A temporal file path to be used for each state transition.
            Usage is same with form_file.

        - proc (subprocess.Popen): A running process.

        - th_oh (OutputHandler): An output handler for subprocess.
    """

    def __init__(
        self,
        config: Config,
        options: dict,
        scheduler: Union[AbciScheduler, LocalScheduler],
        trial_id: int
    ) -> None:
        """Initial method for Job class.

        Args:
            config (ConfileWrapper): A configuration object.
            scheduler (Union[LocalScheduler, AbciScheduler]): A reference for
                scheduler object.
            hp_file (Path): A hyper parameter file for this job.
        """
        super(Job, self).__init__()
        # === Load config file===
        self.config = config
        self.config_path = self.config.config_path
        self.options = options
        # === Get config parameter values ===
        self.workspace = self.config.workspace.get()
        self.cancel_retry = self.config.cancel_retry.get()
        self.cancel_timeout = self.config.cancel_timeout.get()
        self.expire_retry = self.config.expire_retry.get()
        self.expire_timeout = self.config.expire_timeout.get()
        self.finished_retry = self.config.finished_retry.get()
        self.finished_timeout = self.config.finished_timeout.get()
        self.job_loop_duration = self.config.job_loop_duration.get()
        self.job_retry = self.config.job_retry.get()
        self.job_timeout = self.config.job_timeout.get()
        self.kill_retry = self.config.kill_retry.get()
        self.kill_timeout = self.config.kill_timeout.get()
        self.result_retry = self.config.result_retry.get()
        self.batch_job_timeout = self.config.batch_job_timeout.get()
        self.runner_retry = self.config.runner_retry.get()
        self.runner_timeout = self.config.runner_timeout.get()
        self.running_retry = self.config.running_retry.get()
        self.running_timeout = self.config.running_timeout.get()
        self.resource_type = self.config.resource_type.get()

        self.threshold_timeout = None
        self.threshold_retry = None
        self.count_retry = 0

        self.ws = Path(self.workspace).resolve()
        self.abci_output_path = self.ws / aiaccel.dict_output
        self.dict_lock = self.ws / aiaccel.dict_lock

        self.model = Model()
        self.machine = CustomMachine(
            model=self.model,
            states=JOB_STATES,
            transitions=JOB_TRANSITIONS,
            initial=JOB_STATES[0]['name'],
            auto_transitions=False,
            ordered_transitions=False
        )
        self.loop_count = 0
        self.scheduler = scheduler
        global job_lock
        self.lock = job_lock

        self.config_path = str(self.config_path)
        self.trial_id = trial_id
        self.trial_id_str = TrialId(self.config_path).zero_padding_any_trial_id(self.trial_id)
        self.from_file = None
        self.to_file = None
        self.next_state = None
        self.proc = None
        self.th_oh = None
        self.stop_flag = False
        self.storage = Storage(
            self.ws,
            fsmode=self.options['fs'],
            config_path=self.config_path
        )
        self.content = self.storage.get_hp_dict(self.trial_id)
        self.result_file_path = self.ws / aiaccel.dict_result / (self.trial_id_str + '.hp')
        self.expirable_states = [jt["source"] for jt in JOB_TRANSITIONS if jt["trigger"] == "expire"]

    def get_machine(self) -> CustomMachine:
        """Get a state machine object.

        Returns:
            CustomMachine: A state machine object.
        """
        return self.machine

    def get_model(self) -> Model:
        """Get a state transition model object.

        Returns:
            Model: A state transition model object.
        """
        return self.model

    def get_state(self) -> Machine:
        """Get a current state.

        Returns:
            Model: A current state.
        """
        return self.machine.get_state(self.model.state)

    def get_state_name(self) -> Union[str, Enum]:
        """Get a current state name.

        Returns:
            Union[str, Enum]: A current state name.
        """
        state = self.get_state()
        return state.name

    def is_local(self) -> bool:
        """Is the execution on a local computer or not(ABCI).

        Returns:
            bool: Is the execution on a local computer or not(ABCI).
        """
        return self.resource_type == aiaccel.resource_type_local

    def schedule(self) -> None:
        """Schedule to run this execution.

        Returns:
            None
        """
        self.model.schedule(self)

    def stop(self):
        self.stop_flag = True

    def is_timeout(self) -> bool:
        state = self.machine.get_state(self.model.state)
        now = get_time_now_object()
        if self.threshold_timeout is None:
            return False
        return (
            now >= self.threshold_timeout and
            state.name in self.expirable_states
        )

    def is_exceeded_retry_times_max(self) -> bool:
        state = self.machine.get_state(self.model.state)
        if self.threshold_retry is None:
            return False
        return (
            self.count_retry >= self.threshold_retry and
            state.name in self.expirable_states
        )

    def run(self) -> None:
        """Thread.run method.

        Returns:
            None
        """
        logger = logging.getLogger('root.scheduler.job')
        state = None
        buff = Buffer(['state.name'])
        buff.d['state.name'].set_max_len(2)

        while True:
            with self.lock:
                if self.storage.alive.check_alive('scheduler') is False:
                    logger.info('Scheduler alive state is False')
                    break

            self.loop_count += 1

            state = self.machine.get_state(self.model.state)
            buff.Add("state.name", state.name)
            if (
                buff.d["state.name"].Len == 1 or
                buff.d["state.name"].has_difference()
            ):
                with self.lock:
                    self.storage.jobstate.set_any_trial_jobstate(
                        trial_id=self.trial_id,
                        state=state.name
                    )

            if state.name == 'Success' or 'Failure' in state.name:
                break

            now = get_time_now_object()

            if self.is_timeout():
                logger.debug(
                    'Timeout expire state: {}, now: {}, timeout: {}'
                    .format(state.name, now, self.threshold_timeout)
                )
                self.model.expire(self)

            elif self.is_exceeded_retry_times_max():
                logger.debug(
                    'Retry expire state: {}, count: {}, threshold: {}'
                    .format(state.name, self.count_retry, self.threshold_retry)
                )
                self.model.expire(self)

            elif state.name != 'Scheduling':
                self.model.next(self)

            logger.debug(
                'Running job thread, trial id: {}, loop: {}, state: {}, count retry: {}'
                .format(self.trial_id, self.loop_count, state.name, self.count_retry)
            )

            if self.stop_flag:
                break

            time.sleep(self.job_loop_duration)

        logger.info(
            'Thread finished, trial id: {}, loop: {}, state: {}, count retry: {}'
            .format(
                self.trial_id,
                self.loop_count,
                state if state is None else state.name,
                self.count_retry
            )
        )
