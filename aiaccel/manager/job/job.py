from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from omegaconf.dictconfig import DictConfig
from transitions import Machine
from transitions.extensions.states import Tags, add_state_features

from aiaccel.common import datetime_format
from aiaccel.storage import Storage
from aiaccel.util import Buffer, create_job_script_preamble
from aiaccel.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover
    from aiaccel.manager import AbstractManager, AbstractModel


JOB_STATES = [
    {"name": "ready"},
    {"name": "running"},
    {"name": "finished"},
    {"name": "success"},
    {"name": "failure"},
    {"name": "timeout"},
]


JOB_TRANSITIONS: list[dict[str, str | list[str]]] = [
    {
        "trigger": "next_state",
        "source": "ready",
        "dest": "running",
        "before": "before_running",
        "after": "after_running",
    },
    {
        "trigger": "next_state",
        "source": "running",
        "dest": "finished",
        "conditions": "conditions_job_finished",
        "before": "before_finished",
        "after": "after_finished",
    },
    {
        "trigger": "next_state",
        "source": "finished",
        "dest": "success",
    },
    {
        "trigger": "expire",
        "source": ["ready", "running", "finished"],
        "dest": "failure",
    },
    {
        "trigger": "timeout",
        "source": ["ready", "running", "finished"],
        "dest": "timeout",
        "before": "before_timeout",
        "after": "after_timeout",
    },
]


@add_state_features(Tags)
class CustomMachine(Machine):
    pass


class Job:
    """A job thread to manage running jobs on local computer or ABCI."""

    def __init__(self, config: DictConfig, manager: AbstractManager, model: AbstractModel, trial_id: int) -> None:
        super(Job, self).__init__()
        # === Load config file===
        self.config = config
        # === Get config parameter values ===
        self.count_retry = 0
        self.logger = logging.getLogger("root.manager.job")
        self.workspace = Workspace(self.config.generic.workspace)
        self.storage = Storage(self.workspace.storage_file_path)
        self.trial_id = trial_id
        self.content = self.storage.get_hp_dict(self.trial_id)
        self.manager = manager
        self.goals: list[str] = self.config.optimize.goal
        self.model = model
        if self.model is None:
            raise ValueError(
                "model is None. "
                "Be sure to specify the model to use in the Job class. "
                "For example, PylocalManager doesn't use model. "
                "Therefore, Job class cannot be used."
            )
        self.machine = CustomMachine(
            model=self.model,
            states=JOB_STATES,
            transitions=JOB_TRANSITIONS,
            initial=JOB_STATES[0]["name"],
            auto_transitions=False,
            ordered_transitions=False,
        )
        self.job_script_preamble = create_job_script_preamble(
            self.config.ABCI.job_script_preamble, self.config.ABCI.job_script_preamble_path
        )
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.trial_id = trial_id
        self.proc: Any = None
        self.th_oh: Any = None
        self.buff = Buffer(["state.name"])
        self.buff.d["state.name"].set_max_len(2)

    def get_state_name(self) -> str | Enum:
        """Get a current state name.

        Returns:
            str | Enum: A current state name.
        """
        state = self.machine.get_state(self.model.state)
        return state.name

    def set_state(self, state: str) -> None:
        """Set a current state.

        Args:
            state (str): A current state.
        """
        self.machine.set_state(state)

    def write_start_time_to_storage(self) -> None:
        """Set a start time."""
        self.start_time = datetime.now()
        _start_time = self.start_time.strftime(datetime_format)
        self.storage.timestamp.set_any_trial_start_time(trial_id=self.trial_id, start_time=_start_time)

    def write_end_time_to_storage(self) -> None:
        """Set an end time."""
        self.end_time = datetime.now()
        _end_time = self.end_time.strftime(datetime_format)
        self.storage.timestamp.set_any_trial_end_time(trial_id=self.trial_id, end_time=_end_time)

    def write_state_to_storage(self, state: str) -> None:
        """Write a current state to the database."""
        self.storage.trial.set_any_trial_state(trial_id=self.trial_id, state=state)

    def write_job_success_or_failed_to_storage(self) -> None:
        """Write a job success or failed to the database."""
        returncode = self.storage.returncode.get_any_trial_returncode(trial_id=self.trial_id)
        end_state = "success"
        if returncode != 0:
            end_state = "failure"
        self.storage.jobstate.set_any_trial_jobstate(trial_id=self.trial_id, state=end_state)

    def get_job_elapsed_time_in_seconds(self) -> float:
        """Get a job elapsed time in seconds.

        Returns:
            float: A job elapsed time in seconds.
        """
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    def is_timeout(self) -> bool:
        """Check if a job is timeout.

        Returns:
            bool: True if a job is timeout.
        """
        if self.start_time is None:
            return False
        elapsed_time = self.get_job_elapsed_time_in_seconds()
        if elapsed_time > self.config.generic.batch_job_timeout:
            return True
        return False

    def main(self) -> None:
        """Thread.run method.

        Returns:
            None
        """

        state = self.machine.get_state(self.model.state)
        if state.name.lower() in ["failure", "success", "timeout"]:
            return
        if self.is_timeout():
            self.model.timeout(self)
            return
        if state.name.lower() == "ready":
            self.model.next_state(self)
        elif state.name.lower() == "running":
            self.model.next_state(self)
        elif state.name.lower() == "finished":
            self.model.next_state(self)
        else:
            ...
        return
