from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from aiaccel import dict_runner
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.process import kill_process
from aiaccel.util.time_tools import get_time_delta, get_time_now_object

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aiaccel.scheduler.job.job import Job


class AbstractModel(object):
    state: str
    expire: Callable[[Any], Any]
    schedule: Callable[[Any], Any]
    next: Callable[[Any], Any]

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

    def conditions_confirmed(self, obj: 'Job') -> bool:
        """State transition of 'conditions_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target file exists or not.
        """
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
        obj.storage.trial.set_any_trial_state(
            trial_id=obj.trial_id,
            state=obj.next_state
        )
        return

    def get_runner_file(self, obj: 'Job') -> Path:
        return obj.ws / dict_runner / f'run_{obj.trial_id_str}.sh'

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
        raise NotImplementedError

    def conditions_runner_confirmed(self, obj: 'Job') -> bool:
        """State transition of 'conditions_runner_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target file exists or not. But always true if local
                execution.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def conditions_job_confirmed(self, obj: 'Job') -> bool:
        """State transition of 'conditions_job_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target job is finished or not.
        """
        for state in obj.scheduler.stats:
            state_trial_id = obj.scheduler.parse_trial_id(state['name'])
            if state_trial_id is not None and obj.trial_id == int(state_trial_id):
                return True
        else:
            # confirm whether the result file exists or not (this means the job finished quickly
            trial_ids = obj.storage.result.get_result_trial_id_list()
            if trial_ids is None:
                return False
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

        result = obj.storage.result.get_any_trial_objective(trial_id=obj.trial_id)
        error = obj.storage.error.get_any_trial_error(trial_id=obj.trial_id)
        content = obj.storage.get_hp_dict(trial_id=obj.trial_id)
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
            state_trial_id = obj.scheduler.parse_trial_id(state['name'])
            if state_trial_id is not None and obj.trial_id == int(state_trial_id):
                kill_process(state['job-ID'])
        else:
            obj.logger.warning(f'Not matched job trial_id: {obj.trial_id}')

    def conditions_kill_confirmed(self, obj: 'Job') -> bool:
        """State transition of 'conditions_kill_confirmed'.

        Check the details of 'JOB_STATES' and 'JOB_TRANSITIONS'.

        Args:
            obj (Job): A job object.

        Returns:
            bool: A target is killed or not.
        """
        for state in obj.scheduler.stats:
            state_trial_id = obj.scheduler.parse_trial_id(state['name'])
            if state_trial_id is not None and obj.trial_id == int(state_trial_id):
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

        if (
            obj.storage.is_ready(obj.trial_id) or
            obj.storage.is_running(obj.trial_id)
        ):
            obj.storage.trial.set_any_trial_state(trial_id=obj.trial_id, state='ready')
        else:
            obj.logger.warning(f'Could not find any trial_id: {obj.trial_id}')

        obj.threshold_timeout = (
            get_time_now_object() + get_time_delta(obj.expire_timeout)
        )
        obj.threshold_retry = obj.expire_retry

    def change_state(self, obj: 'Job') -> None:
        obj.storage.trial.set_any_trial_state(trial_id=obj.trial_id, state=obj.next_state)
