from __future__ import annotations

from aiaccel.util.process import OutputHandler, exec_runner
from aiaccel.wrapper_tools import create_runner_command
from aiaccel.scheduler.job.model.abstract_model import AbstractModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiaccel.scheduler.job.job import Job


class LocalModel(AbstractModel):

    def before_runner_create(self, obj: 'Job') -> None:
        return None

    def conditions_runner_confirmed(self, obj: 'Job') -> bool:
        return True

    def before_job_submitted(self, obj: 'Job') -> None:
        runner_command = create_runner_command(
            obj.config.generic.job_command,
            obj.content,
            str(obj.trial_id),
            obj.config.config_path
        )

        obj.proc = exec_runner(runner_command)

        obj.th_oh = OutputHandler(
            obj.scheduler,
            obj.proc,
            'Job'
        )
        obj.th_oh.start()
