from __future__ import annotations

from aiaccel.wrapper_tools import create_runner_command
from aiaccel.scheduler import Job
from aiaccel.scheduler.job.model import LocalModel
from aiaccel.experimental.mpi.config import MpiConfig
from aiaccel.experimental.mpi.util.mpi import Mpi, MpiOutputHandler


class MpiModel(LocalModel):

    def before_job_submitted(self, obj: Job) -> None:
        runner_command = create_runner_command(
            obj.config.job_command.get(),
            obj.content,
            obj.trial_id,
            str(obj.config_path),
            str(obj.command_error_output)
        )

        obj.logger.info(f'runner command: {" ".join(runner_command)}')

        gpu_mode = MpiConfig(obj.config.config_path).mpi_gpu_mode.get()
        (processor, tag) = Mpi.submit(
            runner_command,
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
