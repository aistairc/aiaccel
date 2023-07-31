from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aiaccel.util import file_create


def create_abci_batch_file(
    trial_id: int,
    param_content: dict[str, Any],
    output_file_path: Path | str,
    error_file_path: Path | str,
    config_file_path: Path | str,
    batch_file: Path,
    job_script_preamble: Path | str | None,
    command: str,
    enabled_variable_name_argumentation: bool,
    dict_lock: Path,
) -> None:
    """Create a ABCI batch file.

    The 'job_script_preamble' is a base of the ABCI batch file. At first, loads
    'job_script_preamble', and adds the 'commands' to the loaded contents. Finally,
    writes the contents to 'batch_file'.

    Args:
        trial_id (int): A trial id.
        param_content (dict): A dictionary of parameters.
        output_file_path (Path | str): A path of a output file.
        error_file_path (Path | str): A path of a error file.
        config_file_path (Path | str): A path of a config file.
        batch_file (Path): A path of a creating file.
        job_script_preamble (str): A wrapper file of ABCI batch file.
        command (str): A command to execute.
        dict_lock (Path): A directory to store lock files.

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

    set_retult = _generate_command_line(
        command="aiaccel-set-result",
        args=[
            "--file=$output_file_path",
            "--trial_id=$trial_id",
            "--config=$config_file_path",
            "--start_time=$start_time",
            "--end_time=$end_time",
            "--objective=$ys",
            "--error=$error",
            "--exitcode=$exitcode",
            _generate_param_args(param_content["parameters"]),
        ],
    )

    set_retult_no_error = _generate_command_line(
        command="aiaccel-set-result",
        args=[
            "--file=$output_file_path",
            "--trial_id=$trial_id",
            "--config=$config_file_path",
            "--start_time=$start_time",
            "--end_time=$end_time",
            "--objective=$ys",
            "--exitcode=$exitcode",
            _generate_param_args(param_content["parameters"]),
        ],
    )

    main_parts = [
        f"trial_id={str(trial_id)}",
        f"config_file_path={str(config_file_path)}",
        f"output_file_path={str(output_file_path)}",
        f"error_file_path={str(error_file_path)}",
        'start_time=`date "+%Y-%m-%d %H:%M:%S"`',
        f'result=$({" ".join(commands)} | tail -n 1)',
        "exitcode=$?",
        'ys=$(echo $result | tr -d "[],")',
        "error=`cat $error_file_path`",
        'end_time=`date "+%Y-%m-%d %H:%M:%S"`',
        'if [ -n "$error" ]; then',
        "\t" + set_retult,
        "else",
        "\t" + set_retult_no_error,
        "fi",
    ]

    script = ""

    # preamble
    if job_script_preamble is not None:
        with open(job_script_preamble, "r") as f:
            lines = f.read().splitlines()
            if len(lines) > 0:
                for line in lines:
                    script += line + "\n"

    script += "\n"

    # parameters
    for param in param_content["parameters"]:
        if "parameter_name" in param.keys() and "value" in param.keys():
            script += f'{param["parameter_name"]}={param["value"]}' + "\n"

    script += "\n"

    # main
    for s in main_parts:
        script += s + "\n"

    file_create(batch_file, script, dict_lock)


def _generate_command_line(command: str, args: list[str]) -> str:
    return f'{command} {" ".join(args)}'


def _generate_param_args(params: list[dict[str, Any]]) -> str:
    param_args = ""
    for param in params:
        if "parameter_name" in param.keys() and "value" in param.keys():
            param_args += f'--{param["parameter_name"]}=${param["parameter_name"]} '
    return param_args
