from __future__ import annotations

from subprocess import PIPE, Popen, run
from typing import TYPE_CHECKING

from aiaccel.scheduler.job.model.abstract_model import AbstractModel
from aiaccel.util.process import OutputHandler
#from aiaccel.util.process import exec_runner
from aiaccel.wrapper_tools import create_runner_command

if TYPE_CHECKING:
    from aiaccel.scheduler.job.job import Job


class LocalModel(AbstractModel):

    def before_runner_create(self, obj: 'Job') -> None:
        return None

    def conditions_runner_confirmed(self, obj: 'Job') -> bool:
        return True

    def before_job_submitted(self, obj: 'Job') -> None:
        runner_command = create_runner_command(
            obj.config.job_command.get(),
            obj.content,
            obj.trial_id,
            str(obj.config_path),
            str(obj.command_error_output)
        )
        obj.logger.info(f'runner command: {" ".join(runner_command)}')
        obj.proc = Popen(runner_command, stdout=PIPE, stderr=PIPE)

        obj.th_oh = OutputHandler(obj.proc)
        obj.th_oh.start()

    def conditions_result(self, obj: 'Job') -> bool:
        if super().conditions_result(obj):
            return True

        if obj.th_oh.get_returncode() is None:
            return False
        else:
            self.create_result_file(obj)
            return False

    def create_result_file(self, obj: 'Job') -> None:
        trial_id = obj.trial_id
        stdouts = obj.th_oh.get_stdouts()
        stderrs = obj.th_oh.get_stderrs()
        start_time = str(obj.th_oh.get_start_time())
        end_time = str(obj.th_oh.get_end_time())

        objective = stdouts[-1]  # todo
        error = '\n'.join(stderrs)
        output_file_path = str(obj.get_result_file_path())
        config_file_path = str(obj.config_path)

        if len(stderrs) == 0:
            commands = [
                'aiaccel-set-result',
                '--file', output_file_path,
                '--trial_id', str(trial_id),
                '--config', config_file_path,
                '--start_time', start_time,
                '--end_time', end_time,
                '--objective', objective
            ]
        else:
            commands = [
                'aiaccel-set-error',
                '--file', output_file_path,
                '--trial_id', str(trial_id),
                '--config', config_file_path,
                '--start_time', start_time,
                '--end_time', end_time,
                '--objective', objective,
                '--error', error
            ]

        run(commands)

        return None
