from __future__ import annotations

from typing import TYPE_CHECKING

from aiaccel.util import OutputHandler
from aiaccel.util import exec_runner
from aiaccel.wrapper_tools import create_runner_command
from aiaccel.scheduler.job.model import AbstractModel
if TYPE_CHECKING:
    from aiaccel.scheduler import Job


class LocalModel(AbstractModel):

    def before_runner_create(self, obj: Job) -> None:
        return None

    def conditions_runner_confirmed(self, obj: Job) -> bool:
        return True

    def before_job_submitted(self, obj: Job) -> None:
        runner_command = create_runner_command(
            obj.config.job_command.get(),
            obj.content,
            obj.trial_id,
            str(obj.config_path),
            str(obj.command_error_output)
        )

        obj.logger.info(f'runner command: {" ".join(runner_command)}')
        obj.proc = exec_runner(runner_command)

        obj.th_oh = OutputHandler(
            obj.scheduler,
            obj.proc,
            'Job',
            trial_id=obj.trial_id,
            storage=obj.storage
        )
        obj.th_oh.start()
