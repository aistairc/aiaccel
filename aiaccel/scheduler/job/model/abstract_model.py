from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiaccel.scheduler import Job


def _threshold_timeout(timeout_seconds: float) -> datetime:
    return datetime.now() + timedelta(seconds=timeout_seconds)


class AbstractModel(object):
    state: str
    expire: Any
    next_state: Any
    timeout: Any

    def runner_create(self, obj: Job) -> None:  # noqa: U100
        ...

    def job_submitted(self, obj: Job) -> None:  # noqa: U100
        ...

    # running
    def before_running(self, obj: Job) -> None:
        self.runner_create(obj)
        self.job_submitted(obj)
        obj.write_start_time_to_storage()
        obj.write_state_to_storage("running")

    def after_running(self, obj: Job) -> None:  # noqa: U100
        ...

    def conditions_job_finished(self, obj: Job) -> bool:
        objective = obj.storage.result.get_any_trial_objective(trial_id=obj.trial_id)
        return objective is not None

    # finished
    def before_finished(self, obj: Job) -> None:
        obj.write_state_to_storage("finished")
        obj.write_end_time_to_storage()

    def after_finished(self, obj: Job) -> None:
        obj.write_job_success_or_failed_to_storage()

    # timeout  # TODO: implement timeout
    def before_timeout(self, obj: Job) -> None:
        obj.write_state_to_storage("timeout")
        obj.write_end_time_to_storage()

    def after_timeout(self, obj: Job) -> None:
        self.stop_job(obj)

    def stop_job(self, obj: Job) -> None:  # noqa: U100
        ...

    ...
