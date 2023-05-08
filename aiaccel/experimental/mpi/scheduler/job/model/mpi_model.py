from __future__ import annotations

from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any

from aiaccel.wrapper_tools import create_runner_command

from aiaccel.scheduler.job.model import LocalModel

from aiaccel.experimental.mpi.util.mpi import Mpi, MpiOutputHandler

if TYPE_CHECKING:
    from aiaccel.scheduler import Job


class MpiModel(LocalModel):

    def before_job_submitted(self, obj: Job) -> None:
        runner_command = create_runner_command(
            obj.config.generic.job_command,
            obj.content,
            obj.trial_id,
            str(obj.config.config_path),
            str(obj.command_error_output),
        )

        obj.logger.info(f'runner command: {" ".join(runner_command)}')

        gpu_mode = obj.config.resource.mpi_gpu_mode
        (processor, tag) = Mpi.submit(
            runner_command,
            obj.trial_id,
            gpu_mode
        )
        obj.proc = None
        obj.th_oh = MpiOutputHandler(
            obj.scheduler,
            gpu_mode,
            processor,
            tag,
            'Job',
            obj.trial_id,
            storage=obj.storage
        )
        obj.th_oh.start()
        self.is_firsttime_called: bool = False
