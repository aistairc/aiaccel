from aiaccel.util.filesystem import file_create
from pathlib import Path


def create_abci_batch_file(
    batch_file: Path,
    wrapper_file: str,
    commands: list,
    dict_lock: Path
) -> None:
    """Create a ABCI batch file.

    The 'wrapper_file' is a base of the ABCI batch file. At first, loads
    'wrapper_file', and adds the 'commands' to the loaded contents. Finally,
    writes the contents to 'batch_file'.

    Args:
        batch_file (Path): A path of a creating file.
        wrapper_file (str): A wrapper file of ABCI batch file.
        commands (list): Commands to write in a batch file.
        dict_lock (Path): A directory to store lock files.

    Returns:
        None
    """
    with open(wrapper_file, 'r') as f:
        wrapper_lines = f.readlines()

    command_text = ' '.join(commands)
    lines = ''

    for line in wrapper_lines:
        lines += line

    lines += command_text
    file_create(batch_file, lines, dict_lock)
