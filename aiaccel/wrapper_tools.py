from aiaccel.util.filesystem import create_yaml
from pathlib import Path
import aiaccel
import re


def create_runner_command(
    command: str,
    param_content:
    dict,
    hashname: str,
    configpath: str
) -> list:

    """Create a list of command strings to run a hyper parameter.

    Args:
        command (str): A string command.
        param_content (dict): A hyper parameter content.
        hashname (str): A unique name of a hyper parameter.

    Returns:
        A list of command strings.
    """
    commands = re.split(' +', command)
    params = param_content['parameters']
    commands.append('--index')
    commands.append(hashname)
    commands.append('--config')
    commands.append(configpath)

    for param in params:
        # Fix a bug related a negative exponential parameters
        # Need to modify wrapper.py as follows:
        if (
            'parameter_name' in param.keys() and
            'value' in param.keys()
        ):
            commands.append(
                '--{}={}'.format(param['parameter_name'], param['value'])
            )

    return commands


def save_result(
    ws: Path,
    dict_lock: Path,
    hashname: str,
    result: float,
    start_time: str,
    end_time: str,
    err_message: str = ""
) -> None:
    """Save a result file.

    Args:
        ws (Path): A path of a workspace.
        dict_lock (Path): A directory to store lock files.
        hashname (str): An unique name of a parameter set.
        result (float): A result of a parameter set.
        start_time (str): A start time string.
        end_time (str): An end time string.
        err_message (str): Error message from Wrapper (user program)

    Returns:
        None
    """
    result_file = ws / aiaccel.dict_result / '{}.result'.format(hashname)

    contents = {
        'result': result,
        'start_time': start_time,
        'end_time': end_time
    }

    if len(err_message) > 0:
        contents["error"] = err_message

    create_yaml(
        result_file,
        contents,
        dict_lock
    )
