from __future__ import annotations

from pathlib import Path

from aiaccel.util.filesystem import file_create


def create_abci_batch_file(
    trial_id: int,
    param_content: dict,
    output_file_path: Path | str,
    config_file_path: Path | str,
    batch_file: Path,
    job_script_preamble: str,
    commands: list,
    dict_lock: Path
) -> None:
    """Create a ABCI batch file.

    The 'job_script_preamble' is a base of the ABCI batch file. At first, loads
    'job_script_preamble', and adds the 'commands' to the loaded contents. Finally,
    writes the contents to 'batch_file'.

    Args:
        batch_file (Path): A path of a creating file.
        job_script_preamble (str): A wrapper file of ABCI batch file.
        commands (list): Commands to write in a batch file.
        dict_lock (Path): A directory to store lock files.

    Returns:
        None
    """
    with open(job_script_preamble, 'r') as f:
        wrapper_lines = f.readlines()

    command_text = ' '.join(commands)
    lines = ''

    for line in wrapper_lines:
        lines += line
    
    params = param_content['parameters']
    for param in params:
        if 'parameter_name' in param.keys() and 'value' in param.keys():
            lines += f'export {param["parameter_name"]}={param["value"]}\n'
    
    lines += f'trial_id={trial_id}\n'
    lines += 'start_time=`date "+%Y-%m-%d %H:%M:%S"`\n'

    lines += f'result=`{command_text}`\n' 

    lines += 'end_time=`date "+%Y-%m-%d %H:%M:%S"`\n'

    lines += f'aiaccel-set-result --file {str(output_file_path)} --trial_id {trial_id} \
        --config {str(config_file_path)} --start_time $start_time --end_time $end_time\
        --objective $result --error $error '
    for param in params:
        if 'parameter_name' in param.keys() and 'value' in param.keys():
            lines += f'--{param["parameter_name"]} {param["value"]} '

    file_create(batch_file, lines, dict_lock)
