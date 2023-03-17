from __future__ import annotations

import re
from typing import Any


def create_runner_command(
    command: str,
    param_content: dict[str, Any],
    trial_id: int,
    config_path: str,
    command_error_output: str
) -> list[str]:
    """Create a list of command strings to run a hyper parameter.

    Args:
        command (str): A string command.
        param_content (dict): A hyper parameter content.
        trial_id (str): A unique name of a hyper parameter.

    Returns:
        list[str]: A list of command strings.
    """
    commands = re.split(' +', command)
    params = param_content['parameters']
    for param in params:
        # Fix a bug related a negative exponential parameters
        # Need to modify wrapper.py as follows:
        if 'parameter_name' in param.keys() and 'value' in param.keys():
            commands.append(f'--{param["parameter_name"]}')
            commands.append(f'{param["value"]}')
    commands.append('--trial_id')
    commands.append(str(trial_id))
    commands.append('--config')
    commands.append(config_path)
    commands.append('2>')
    commands.append(command_error_output)
    return commands
