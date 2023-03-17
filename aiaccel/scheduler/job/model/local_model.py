from __future__ import annotations

from subprocess import PIPE, Popen, run
from typing import TYPE_CHECKING

from aiaccel.scheduler.job.model.abstract_model import AbstractModel
from aiaccel.util.process import OutputHandler
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
        """Create result file.

        Args:
            obj (Job): Job object.

        Returns:
            None
        """
        trial_id = obj.trial_id
        stdouts = obj.th_oh.get_stdouts()
        stderrs = obj.th_oh.get_stderrs()
        start_time = str(obj.th_oh.get_start_time())
        end_time = str(obj.th_oh.get_end_time())
        exitcode = str(obj.th_oh.get_returncode())
        params = obj.content['parameters']

        if len(stdouts) == 0:
            objective = 'nan'
        else:
            objective = stdouts[-1]  # TODO: fix
        error = '\n'.join(stderrs)

        args = {
            'workspace': str(obj.workspace.path),
            'trial_id': str(trial_id),
            'start_time': start_time,
            'end_time': end_time,
            'objective': objective,
            'error': error,
            'exitcode': str(exitcode)
        }

        if len(error) == 0:
            del args['error']

        if objective == 'None':
            del args['objective']

        commands = ['aiaccel-set-result']
        for key in args.keys():
            commands.append('--' + key)
            commands.append(str(args[key]))

        for param in params:
            if 'parameter_name' in param.keys() and 'value' in param.keys():
                commands.append('--' + param['parameter_name'])
                commands.append(str(param['value']))

        run(commands)

        return None
