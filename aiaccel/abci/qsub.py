from __future__ import annotations

from pathlib import Path

from aiaccel.common import dict_output
from aiaccel.config import Config


''' Example of stat
stat = {
    'job-ID': 12345,
    'prior': 0.25586,
    'name': 'run.sh',
    'user': 'username',
    'state': 'r',
    'submit/start at': '06/27/2018 21:14:49',
    'queue': 'gpu@g0001',
    'jclass': '',
    'slots': 80,
    'ja-task-ID': ''
}
'''


def create_qsub_command(config: Config, runner_file: Path) -> list[str]:
    """Create ABCI 'qsub' command.

    Args:
        config (Config): A Config object.
        runner_file (Path): A path of 'qsub' batch file.

    Returns:
        list: A list to run 'qsub' command.
    """
    path = Path(config.workspace.get()).resolve()
    job_execution_options = config.job_execution_options.get()

    command = [
        'qsub',
        '-g', f'{config.abci_group.get()}',
        '-j', 'y',
        '-o', f'{path / dict_output}',
        str(runner_file)
    ]

    #
    # additional option
    #

    # no additional option
    command_tmp = command.copy()
    if job_execution_options is None:
        return command
    if job_execution_options == '':
        return command
    if job_execution_options == []:
        return command

    # add option
    if type(job_execution_options) == str:
        for cmd in job_execution_options.split(' '):
            command_tmp.insert(-1, cmd)
    elif type(job_execution_options) == list:
        for option in job_execution_options:
            for cmd in option.split(' '):
                command_tmp.insert(-1, cmd)
    else:
        raise ValueError(
            f"job_execution_options: {job_execution_options} is invalid value"
        )

    return command_tmp
