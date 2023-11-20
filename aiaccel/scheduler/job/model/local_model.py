from __future__ import annotations

import re
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any

from aiaccel.scheduler.job.model.abstract_model import AbstractModel
from aiaccel.util import OutputHandler

if TYPE_CHECKING:
    from aiaccel.scheduler import Job


class LocalModel(AbstractModel):
    def runner_create(self, obj: Job) -> None:  # noqa: U100
        pass

    def job_submitted(self, obj: Job) -> None:
        runner_command = self.create_runner_command(
            obj.config.generic.job_command,
            obj.content,
            obj.trial_id,
            str(obj.config.config_path),
            str(obj.workspace.get_error_output_file(obj.trial_id)),
            obj.config.generic.enabled_variable_name_argumentation,
        )
        obj.logger.info(f'runner command: {" ".join(runner_command)}')
        obj.proc = Popen(runner_command, stdout=PIPE, stderr=PIPE)

        obj.th_oh = OutputHandler(obj.proc)
        obj.th_oh.start()
        self.is_firsttime_called = False

    def conditions_job_finished(self, obj: "Job") -> bool:
        if super().conditions_job_finished(obj):
            return True
        if obj.th_oh.get_returncode() is None or self.is_firsttime_called:
            return False
        else:
            self.write_results_to_database(obj)
            self.is_firsttime_called = True
            return False

    def stop_job(self, obj: Job) -> None:
        obj.th_oh.enforce_kill()

    def create_runner_command(
        self,
        command: str,
        param_content: dict[str, Any],
        trial_id: int,
        config_path: str,
        command_error_output: str,
        enabled_variable_name_argumentation: bool,
    ) -> list[str]:
        """Create a list of command strings to run a hyper parameter.

        Args:
            command (str): A string command.
            param_content (dict): A hyper parameter content.
            trial_id (str): A unique name of a hyper parameter.

        Returns:
            list[str]: A list of command strings.
        """
        commands = re.split(" +", command)
        params = param_content["parameters"]
        if enabled_variable_name_argumentation:
            """
            --name=value
            """
            for param in params:
                if "parameter_name" in param.keys() and "value" in param.keys():
                    commands.append(f'--{param["parameter_name"]}={param["value"]}')
            commands.append(f"--trial_id={str(trial_id)}")
            commands.append(f"--config={config_path}")
        else:
            """
            value
            """
            for param in params:
                if "name" in param.keys() and "value" in param.keys():
                    commands.append(f'{param["value"]}')
            commands.append(str(trial_id))
            commands.append(config_path)
        commands.append("2>")
        commands.append(command_error_output)
        return commands

    def write_results_to_database(self, obj: "Job") -> None:
        """Create result file.

        Args:
            obj (Job): Job object.

        Returns:
            None
        """
        trial_id: str = str(obj.trial_id)
        stdouts: list[str] = obj.th_oh.get_stdouts()
        stderrs: list[str] = obj.th_oh.get_stderrs()
        returncode: int = obj.th_oh.get_returncode()
        params: list[dict[str, Any]] = obj.content["parameters"]
        objective: str = "nan"
        objectives: list[str] = []

        if len(stdouts) > 0:
            if len(stdouts) >= len(obj.goals):
                objectives = stdouts[-len(obj.goals) :]
            elif len(stdouts) == 1:
                objectives.append(stdouts[0])
            elif len(stdouts) > 1:
                for i in range(len(obj.goals)):
                    o_index = len(stdouts) - len(obj.goals) + i
                    objectives.append(stdouts[o_index])
            else:
                raise NotImplementedError("Not Readched")
            if len(stdouts) < len(obj.goals):
                obj.logger.warning(
                    f"Number of objectives is less than the number of goals. "
                    f"Number of objectives: {len(stdouts)}, "
                    f"Number of goals: {len(obj.goals)}"
                )

        error = "\n".join(stderrs)
        args = {
            "storage_file_path": str(obj.workspace.storage_file_path),
            "trial_id": str(trial_id),
            "error": error,
            "returncode": returncode,
        }
        if len(error) == 0:
            del args["error"]

        # commands = ["aiaccel-set-result"]
        commands = ["python", "-m", "aiaccel.cli.set_result"]
        for key in args.keys():
            commands.append(f"--{key}={str(args[key])}")

        commands.append("--objective")
        for objective in objectives:
            commands.append(str(objective))

        for param in params:
            if "name" in param.keys() and "value" in param.keys():
                commands.append(f'--{param["name"]}={param["value"]}')

        obj.logger.debug(f'{" ".join(commands)}')
        Popen(commands)
        return None
