from __future__ import annotations

import re
from pathlib import Path

from aiaccel.util.filesystem import file_create


class JobScrGenerator:
    def __init__(self):
        self.codetxt = ""
        self.indent_level = 0

    def new_line(self):
        self.codetxt += "\n"

    def add_line(self, line: str):
        for _ in range(self.indent_level):
            self.codetxt += "    "
        self.codetxt += line + "\n"

    def add_lines(self, lines: list[str]):
        for line in lines:
            self.add_line(line)

    def indent(self):
        self.indent_level += 1

    def unindent(self):
        self.indent_level -= 1

    def reset_indent_level(self):
        self.indent_level = 0

    def get_code(self):
        return self.codetxt


def create_abci_batch_file(
    trial_id: int,
    param_content: dict,
    output_file_path: Path | str,
    error_file_path: Path | str,
    config_file_path: Path | str,
    batch_file: Path,
    job_script_preamble: Path | str | None,
    command: str,
    dict_lock: Path
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

    commands = re.split(' +', command)
    # for key in param_content.keys():
    #     commands.append(f'--{key}')
    #     commands.append(f'${key}')

    for param in param_content['parameters']:
        if 'parameter_name' in param.keys() and 'value' in param.keys():
            commands.append(f'--{param["parameter_name"]}')
            commands.append(f'${param["parameter_name"]}')
    commands.append('--trial_id')
    commands.append(str(trial_id))
    commands.append('--config')
    commands.append('$config_file_path')
    commands.append('2>')
    commands.append('$error_file_path')

    code = JobScrGenerator()

    if job_script_preamble is not None:
        with open(job_script_preamble, 'r') as f:
            code.add_lines(f.read().splitlines())
    else:
        code.add_line('#!/bin/bash')

    code.new_line()

    for param in param_content['parameters']:
        if 'parameter_name' in param.keys() and 'value' in param.keys():
            code.add_line(f'{param["parameter_name"]}={param["value"]}')
    code.add_line(f'trial_id={trial_id}')
    code.add_line(f'config_file_path={str(config_file_path)}')
    code.add_line(f'output_file_path={str(output_file_path)}')
    code.add_line(f'error_file_path={str(error_file_path)}')
    code.add_line('start_time=`date "+%Y-%m-%d %H:%M:%S"`')
    code.add_line(f'result=`{" ".join(commands)}`')
    code.add_line('error=`cat $error_file_path`')
    code.add_line('end_time=`date "+%Y-%m-%d %H:%M:%S"`')

    code.add_line('if [ -n "$error" ]; then')
    code.indent()
    code.add_line(_generate_command_line(
        command='aiaccel-set-result',
        args=[
            '--file $output_file_path',
            '--trial_id $trial_id',
            '--config $config_file_path',
            '--start_time $start_time',
            '--end_time $end_time',
            '--objective $result',
            '--error $error',
            _generate_param_args(param_content['parameters'])
        ])
    )
    code.unindent()
    code.add_line('else')
    code.indent()
    code.add_line(_generate_command_line(
        command='aiaccel-set-result',
        args=[
            '--file $output_file_path',
            '--trial_id $trial_id',
            '--config $config_file_path',
            '--start_time $start_time',
            '--end_time $end_time',
            '--objective $result',
            _generate_param_args(param_content['parameters'])
        ])
    )
    code.unindent()
    code.add_line('fi')
    code.new_line()

    file_create(batch_file, code.get_code(), dict_lock)


def _generate_command_line(command: str, args: list[str]) -> str:
    return f'{command} {" ".join(args)}'


def _generate_param_args(params: list[dict]) -> str:
    param_args = ''
    for param in params:
        if 'parameter_name' in param.keys() and 'value' in param.keys():
            param_args += f'--{param["parameter_name"]} ${param["parameter_name"]} '
    return param_args
