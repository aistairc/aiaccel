from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from transitions import Machine
from transitions.extensions.states import Tags, add_state_features

from aiaccel.scheduler.job.model.abci_model import AbciModel
from aiaccel.scheduler.job.model.local_model import LocalModel
from aiaccel.util.buffer import Buffer
from aiaccel.util.time_tools import get_time_now_object
from aiaccel.util.trialid import TrialId
from aiaccel.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover
    from aiaccel.scheduler.abci_scheduler import AbciScheduler
    from aiaccel.scheduler.local_scheduler import LocalScheduler

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
        'before': 'before_result',
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


@add_state_features(Tags)
class CustomMachine(Machine):
    pass


class Job:
    """A job thread to manage running jobs on local computer or ABCI.

    Todo:
        Confirm the state transition especially timeout expire and retry
        expire. Retry expire works well?

    Args:
        config (ConfileWrapper): A configuration object.
        scheduler (LocalScheduler | AbciScheduler): A reference for
            scheduler object.
        model (LocalModel | AbciModel): A reference for
            model object of state machine.
        hp_file (Path): A hyper parameter file for this job.

    Raises:
        ValueError: When model is None.

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

        - scheduler (LocalScheduler | AbciScheduler):
            A reference for scheduler object.

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
        scheduler: AbciScheduler | LocalScheduler,
        model: AbciModel | LocalModel,
        trial_id: int
    ) -> None:
        super(Job, self).__init__()
        # === Load config file===
        self.config = config
        self.config_path = self.config.config_path
        # === Get config parameter values ===
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

        self.workspace = Workspace(self.config.workspace.get())

        self.scheduler = scheduler
        self.model = model
        if self.model is None:
            raise ValueError(
                "model is None. "
                "Be sure to specify the model to use in the Job class. "
                "For example, PylocalScheduler doesn't use model. "
                "Therefore, Job class cannot be used."
            )

        self.machine = CustomMachine(
            model=self.model,
            states=JOB_STATES,
            transitions=JOB_TRANSITIONS,
            initial=JOB_STATES[0]['name'],
            auto_transitions=False,
            ordered_transitions=False
        )
        self.loop_count = 0

        self.config_path = str(self.config_path)
        self.trial_id = trial_id
        self.trial_id_str = TrialId(self.config_path).zero_padding_any_trial_id(self.trial_id)
        self.from_file = None
        self.to_file = None
        self.next_state = None
        self.proc = None
        self.th_oh = None
        self.stop_flag = False
        self.storage = Storage(self.workspace.path)
        self.content = self.storage.get_hp_dict(self.trial_id)
        self.result_file_path = self.workspace.result / (self.trial_id_str + '.hp')
        self.expirable_states = [jt["source"] for jt in JOB_TRANSITIONS if jt["trigger"] == "expire"]

        self.buff = Buffer(['state.name'])
        self.buff.d['state.name'].set_max_len(2)

        self.logger = logging.getLogger('root.scheduler.job')

        self.command_error_output = self.workspace.error / f'{self.trial_id}.txt'

    def get_machine(self) -> CustomMachine:
        """Get a state machine object.

        Returns:
            CustomMachine: A state machine object.
        """
        return self.machine

    def get_model(self) -> AbciModel | LocalModel:
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

    def get_state_name(self) -> str | Enum:
        """Get a current state name.

        Returns:
            str | Enum: A current state name.
        """
        state = self.get_state()
        return state.name

    def schedule(self) -> None:
        """Schedule to run this execution.

        Returns:
            None
        """
        self.model.schedule(self)

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

    def check_job_command_error(self) -> bool:
        if self.command_error_output.exists() is False:
            return True

        err = self.command_error_output.read_text()
        if len(err) > 0:
            self.storage.error.set_any_trial_error(trial_id=self.trial_id, error_message=err)
            return False

        return True

    def get_result_file_path(self) -> Path:
        """Get a path to the result file.

        Args:
            trial_id (int): Trial Id.

        Returns:
            PosixPath: A Path object which points to the result file.
        """
        return self.workspace.get_any_result_file_path(trial_id=self.trial_id)

    def main(self) -> None:
        """Thread.run method.

        Returns:
            None
        """

        state = self.machine.get_state(self.model.state)

        self.buff.Add("state.name", state.name)
        if (
            self.buff.d["state.name"].Len == 1 or
            self.buff.d["state.name"].has_difference()
        ):
            self.storage.jobstate.set_any_trial_jobstate(trial_id=self.trial_id, state=state.name)

        if state.name.lower() == 'success':
            return

        if 'failure' in state.name.lower():
            if self.storage.error.get_any_trial_error(trial_id=self.trial_id) is None:
                self.storage.error.set_any_trial_error(
                    trial_id=self.trial_id,
                    error_message=state.name
                )
            return

        if self.is_timeout():
            self.logger.debug(
                f'Timeout expire state: {state.name}, '
                f'now: {get_time_now_object()}, '
                f'timeout: {self.threshold_timeout}'
            )
            self.model.expire(self)

        elif self.is_exceeded_retry_times_max():
            self.logger.debug(
                f'Retry expire state: {state.name}, '
                f'count: {self.count_retry}, '
                f'threshold: {self.threshold_retry}'
            )
            self.model.expire(self)

        elif state.name.lower() != 'scheduling':
            self.model.next(self)

        self.logger.debug(
            f'Running job, '
            f'trial id: {self.trial_id}, '
            f'state: {state.name} ,'
            f'count retry: {self.count_retry}'
        )

        self.check_job_command_error()

        return
