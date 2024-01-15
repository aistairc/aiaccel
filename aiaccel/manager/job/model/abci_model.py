from __future__ import annotations

import re
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any

import fasteners
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from aiaccel.common import dict_stderr, dict_stdout
from aiaccel.manager.job.model.abstract_model import AbstractModel
from aiaccel.util import OutputHandler

if TYPE_CHECKING:
    from aiaccel.manager.job import Job


class AbciModel(AbstractModel):
    def runner_create(self, obj: Job) -> None:
        runner_file_path = obj.workspace.get_runner_file(obj.trial_id)
        self.create_abci_batch_file(
            trial_id=obj.trial_id,
            param_content=obj.content,
            storage_file_path=obj.workspace.storage_file_path,
            error_file_path=obj.workspace.get_error_output_file(obj.trial_id),
            config_file_path=obj.config.config_path,
            runner_file_path=runner_file_path,
            job_script_preamble=obj.job_script_preamble,
            command=obj.config.generic.job_command,
            enabled_variable_name_argumentation=obj.config.generic.enabled_variable_name_argumentation,
            dict_lock=obj.workspace.lock,
        )

    def job_submitted(self, obj: Job) -> None:
        runner_file_path = obj.workspace.get_runner_file(obj.trial_id)
        runner_command = self.create_qsub_command(obj.config, runner_file_path)

        obj.logger.info(f'runner command: {" ".join(runner_command)}')
        obj.proc = Popen(runner_command, stdout=PIPE, stderr=PIPE, bufsize=0)

        obj.th_oh = OutputHandler(obj.proc)
        obj.th_oh.start()

    def generate_command_line(self, command: str, args: list[str]) -> str:
        return f'{command} {" ".join(args)}'

    def generate_param_args(self, params: list[dict[str, Any]]) -> str:
        param_args = ""
        for param in params:
            if "parameter_name" in param.keys() and "value" in param.keys():
                param_args += f'--{param["parameter_name"]}=${param["parameter_name"]} '
        return param_args

    def create_abci_batch_file(
        self,
        trial_id: int,
        param_content: dict[str, Any],
        storage_file_path: Path | str,
        error_file_path: Path | str,
        config_file_path: Path | str,
        runner_file_path: Path,
        job_script_preamble: str,
        command: str,
        enabled_variable_name_argumentation: bool,
        dict_lock: Path,
    ) -> None:
        """Create a ABCI batch file.

        The 'job_script_preamble' is a base of the ABCI batch file. At first, loads
        'job_script_preamble', and adds the 'commands' to the loaded contents. Finally,
        writes the contents to 'runner_file_path'.

        Args:
            -
        Returns:
            None
        """

        commands = re.split(" +", command)
        if enabled_variable_name_argumentation:
            for param in param_content["parameters"]:
                if "parameter_name" in param.keys() and "value" in param.keys():
                    commands.append(f'--{param["parameter_name"]}=${param["parameter_name"]}')
            commands.append(f"--trial_id={str(trial_id)}")
            commands.append("--config=$config_file_path")
        else:
            for param in param_content["parameters"]:
                if "parameter_name" in param.keys() and "value" in param.keys():
                    commands.append(f'${param["parameter_name"]}')
            commands.append(str(trial_id))
            commands.append("$config_file_path")
        commands.append("2>")
        commands.append("$error_file_path")

        set_retult = self.generate_command_line(
            command="python -m aiaccel.cli.set_result",
            args=[
                "--storage_file_path=$storage_file_path",
                "--trial_id=$trial_id",
                "--config=$config_file_path",
                "--objective=$ys",
                "--error=$error",
                "--returncode=$returncode",
                self.generate_param_args(param_content["parameters"]),
            ],
        )

        set_retult_no_error = self.generate_command_line(
            command="python -m aiaccel.cli.set_result",
            args=[
                "--storage_file_path=$storage_file_path",
                "--trial_id=$trial_id",
                "--config=$config_file_path",
                "--objective=$ys",
                "--returncode=$returncode",
                self.generate_param_args(param_content["parameters"]),
            ],
        )

        main_parts = [
            f"trial_id={str(trial_id)}",
            f"config_file_path={str(config_file_path)}",
            f"storage_file_path={str(storage_file_path)}",
            f"error_file_path={str(error_file_path)}",
            f'result=`{" ".join(commands)}`',
            "returncode=$?",
            'ys=$(echo $result | tr -d "[],")',
            "error=`cat $error_file_path`",
            'if [ -n "$error" ]; then',
            "\t" + set_retult,
            "else",
            "\t" + set_retult_no_error,
            "fi",
        ]

        script = ""
        # preamble
        if job_script_preamble is not None and job_script_preamble != "":
            script += job_script_preamble + "\n"
        script += "\n"
        # parameters
        for param in param_content["parameters"]:
            if "parameter_name" in param.keys() and "value" in param.keys():
                script += f'{param["parameter_name"]}={param["value"]}' + "\n"
        script += "\n"
        # main
        for s in main_parts:
            script += s + "\n"
        self.file_create(runner_file_path, script, dict_lock)

    def file_create(self, path: Path, content: str, dict_lock: Path | None = None) -> None:
        """Create a text file.
        Args:
            path (Path): The path of the created file.
            content (str): The content of the created file.
            dict_lock (Path | None, optional): The path to store lock files.
                Defaults to None.

        Returns:
            None
        """
        if dict_lock is None:
            with open(path, "w") as f:
                f.write(content)
        else:
            lock_file = dict_lock / f"{path.parent.name}.lock"
            with fasteners.InterProcessLock(lock_file):
                with open(path, "w") as f:
                    f.write(content)

    def create_qsub_command(self, config: DictConfig, runner_file: Path) -> list[str]:
        """Create ABCI 'qsub' command.

        Args:
            config (Config): A Config object.
            runner_file (Path): A path of 'qsub' batch file.

        Returns:
            list: A list to run 'qsub' command.
        """
        path = Path(config.generic.workspace).resolve()
        job_execution_options = config.ABCI.job_execution_options

        command = [
            "qsub",
            "-g",
            f"{config.ABCI.group}",
            "-o",
            f"{path / dict_stdout}",
            "-e",
            f"{path / dict_stderr}",
            str(runner_file),
        ]

        #
        # additional option
        #

        # no additional option
        command_tmp = command.copy()
        if job_execution_options is None:
            return command
        if job_execution_options == "":
            return command
        if job_execution_options == []:
            return command

        # add option
        if isinstance(job_execution_options, str):
            for cmd in job_execution_options.split(" "):
                command_tmp.insert(-1, cmd)
        elif isinstance(job_execution_options, list) or isinstance(job_execution_options, ListConfig):
            for option in job_execution_options:
                for cmd in option.split(" "):
                    command_tmp.insert(-1, cmd)
        else:
            raise ValueError(f"job_execution_options: {job_execution_options} is invalid value")

        return command_tmp
