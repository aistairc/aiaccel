from __future__ import annotations

from subprocess import Popen
from typing import TYPE_CHECKING, Any

from aiaccel.experimental.mpi.util.mpi import Mpi, MpiOutputHandler
from aiaccel.scheduler.job.model import LocalModel
from aiaccel.wrapper_tools import create_runner_command

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
            obj.config.generic.enable_command_argument
        )

        obj.logger.info(f'runner command: {" ".join(runner_command)}')

        gpu_mode = obj.config.resource.mpi_gpu_mode
        (processor, tag) = Mpi.submit(runner_command, obj.trial_id, gpu_mode)
        obj.proc = None
        obj.th_oh = MpiOutputHandler(obj.scheduler, gpu_mode, processor, tag, "Job", obj.trial_id, storage=obj.storage)
        obj.th_oh.start()
        self.is_firsttime_called = False

    def conditions_result(self, obj: "Job") -> bool:
        if super().conditions_result(obj):
            return True

        if obj.th_oh.get_returncode() is None or self.is_firsttime_called:
            return False
        else:
            self.create_result_file(obj)
            self.is_firsttime_called = True
            return False

    def create_result_file(self, obj: "Job") -> None:
        """Create result file.

        Args:
            obj (Job): Job object.

        Returns:
            None
        """
        trial_id: str = str(obj.trial_id)
        stdouts: list[str] = obj.th_oh.get_stdouts()
        stderrs: list[str] = obj.th_oh.get_stderrs()
        start_time: str = str(obj.th_oh.get_start_time())
        end_time: str = str(obj.th_oh.get_end_time())
        exitcode: str = str(obj.th_oh.get_returncode())
        params: list[dict[str, Any]] = obj.content["parameters"]
        objective: str = "nan"
        objectives: list[str] = []

        if len(stdouts) > 0:
            objective = stdouts[-1]  # TODO: fix
            objective = objective.strip("[]")
            objective = objective.replace(" ", "")
            objectives = objective.split(",")

        error = "\n".join(stderrs)
        output_file_path = str(obj.get_result_file_path())
        config_file_path = str(obj.config.config_path)

        args = {
            "file": output_file_path,
            "trial_id": trial_id,
            "config": config_file_path,
            "start_time": start_time,
            "end_time": end_time,
            "error": error,
            "exitcode": exitcode,
        }

        if len(error) == 0:
            del args["error"]

        commands = ["python", "-m", "aiaccel.experimental.mpi.cli.set_result"]
        for key in args.keys():
            commands.append(f"--{key}={str(args[key])}")

        commands.append("--objective")
        for objective in objectives:
            commands.append(str(objective))

        for param in params:
            if "parameter_name" in param.keys() and "value" in param.keys():
                commands.append(f"--{param['parameter_name']}={str(param['value'])}")
        print(commands)
        Popen(commands)

        return None
