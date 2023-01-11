from pathlib import Path

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from aiaccel import dict_output

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


def create_qsub_command(config: DictConfig, runner_file: Path) -> list:
    """Create ABCI 'qsub' command.

    Args:
        config (ConfileWrapper): A configuration object.
        runner_file (Path): A path of 'qsub' batch file.

    Returns:
        list: A list to run 'qsub' command.
    """
    path = Path(config.generic.workspace).resolve()
    job_execution_options = config.ABCI.job_execution_options

    command = [
        'qsub',
        '-g', f'{config.ABCI.group}',
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
    elif type(job_execution_options) == list or type(job_execution_options) == ListConfig:
        for option in job_execution_options:
            for cmd in option.split(' '):
                command_tmp.insert(-1, cmd)
    else:
        raise ValueError(
            f"job_execution_options: {job_execution_options} is invalid value"
        )

    return command_tmp
