from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aiaccel.common import dict_result
from aiaccel.util import create_yaml


def create_runner_command(
    command: str,
    param_content: dict[str, Any],
    trial_id: int,
    config_path: str,
    command_error_output: str,
    enable_name_in_optional_argument: bool,
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
    if enable_name_in_optional_argument:
        for param in params:
            if "parameter_name" in param.keys() and "value" in param.keys():
                commands.append(f'--{param["parameter_name"]}={param["value"]}')
        commands.append(f"--trial_id={str(trial_id)}")
        commands.append(f"--config={config_path}")
    else:
        for param in params:
            if "parameter_name" in param.keys() and "value" in param.keys():
                commands.append(f'{param["value"]}')
        commands.append(str(trial_id))
        commands.append(config_path)
    commands.append("2>")
    commands.append(command_error_output)
    return commands


def save_result(
    ws: Path, dict_lock: Path, trial_id_str: str, result: float, start_time: str, end_time: str, err_message: str = ""
) -> None:
    """Save a result file.

    Args:
        ws (Path): A path of a workspace.
        dict_lock (Path): A directory to store lock files.
        trial_id_str (str): An unique name of a parameter set.
        result (float): A result of a parameter set.
        start_time (str): A start time string.
        end_time (str): An end time string.
        err_message (str): Error message from Wrapper (user program)

    Returns:
        None
    """
    result_file = ws / dict_result / f"{trial_id_str}.result"

    contents = {"result": result, "start_time": start_time, "end_time": end_time}

    if len(err_message) > 0:
        contents["error"] = err_message

    create_yaml(result_file, contents, dict_lock)
