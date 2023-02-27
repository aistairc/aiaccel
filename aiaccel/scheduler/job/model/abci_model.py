from __future__ import annotations

from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING

import fasteners

from aiaccel import dict_result, extension_hp
from aiaccel.abci.batch import create_abci_batch_file
from aiaccel.abci.qsub import create_qsub_command
from aiaccel.scheduler.job.model.abstract_model import AbstractModel
from aiaccel.util.filesystem import interprocess_lock_file
from aiaccel.util.process import OutputHandler, exec_runner
from aiaccel.util.retry import retry
from aiaccel.wrapper_tools import create_runner_command

if TYPE_CHECKING:
    from aiaccel.scheduler.job.job import Job


class AbciModel(AbstractModel):
    def before_runner_create(self, obj: 'Job') -> None:
        # commands = create_runner_command(
        #     obj.config.job_command.get(),
        #     obj.content,
        #     obj.trial_id,
        #     str(obj.config_path),
        #     str(obj.command_error_output)
        # )

        create_abci_batch_file(
            trial_id=obj.trial_id,
            param_content=obj.content,
            output_file_path=obj.get_result_file_path(),
            error_file_path=obj.command_error_output,
            config_file_path=obj.config_path,
            batch_file=obj.to_file,
            job_script_preamble=obj.config.job_script_preamble.get(),
            command=obj.config.job_command.get(),
            dict_lock=obj.dict_lock
        )

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def conditions_runner_confirmed(self, obj: 'Job') -> bool:
        lockpath = interprocess_lock_file(obj.to_file, obj.dict_lock)
        with fasteners.InterProcessLock(lockpath):
            return obj.to_file.exists()

    def before_job_submitted(self, obj: 'Job') -> None:
        runner_file = self.get_runner_file(obj)
        runner_command = create_qsub_command(
            obj.config,
            str(runner_file)
        )

        obj.logger.info(f'runner command: {" ".join(runner_command)}')
        obj.proc = Popen(runner_command, stdout=PIPE, stderr=PIPE)

        obj.th_oh = OutputHandler(obj.proc)
        obj.th_oh.start()
