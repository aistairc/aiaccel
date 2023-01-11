from __future__ import annotations


import fasteners

from aiaccel.abci.batch import create_abci_batch_file
from aiaccel.abci.qsub import create_qsub_command
from aiaccel.util.filesystem import interprocess_lock_file
from aiaccel.util.process import OutputHandler, exec_runner
from aiaccel.util.retry import retry
from aiaccel.wrapper_tools import create_runner_command
from aiaccel.scheduler.job.model.abstract_model import AbstractModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiaccel.scheduler.job.job import Job


class AbciModel(AbstractModel):

    def before_runner_create(self, obj: 'Job') -> None:
        commands = create_runner_command(
            obj.config.generic.job_command,
            obj.content,
            obj.trial_id,
            str(obj.config.config_path),
            str(obj.command_error_output)
        )

        create_abci_batch_file(
            obj.to_file,
            obj.config.ABCI.job_script_preamble,
            commands,
            obj.dict_lock
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
        obj.proc = exec_runner(runner_command)

        obj.th_oh = OutputHandler(
            obj.scheduler,
            obj.proc,
            'Job',
            trial_id=obj.trial_id,
            storage=obj.storage
        )
        obj.th_oh.start()
