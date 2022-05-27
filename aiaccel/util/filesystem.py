from pathlib import Path
from typing import List
import aiaccel
import fasteners
import shutil
import yaml
from functools import wraps
import time


def retry(_MAX_NUM=60, _DELAY=1.0):
    def _retry(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            for i in range(_MAX_NUM):
                if i == _MAX_NUM - 1:
                    raise
                try:
                    return func(*args, **kwargs)
                except BaseException:
                    time.sleep(_DELAY)
                    continue
        return _wrapper
    return _retry


def check_alive_file(path: Path, dict_lock: Path = None) -> bool:
    """Check whether the alive file exists or not.

    Args:
        path (Path): The path to the alive file.
        dict_lock (Path): The directory to store lock files.

    Returns:
        bool: Whether the alive file exists or not.
    """
    if dict_lock is None:
        return path.exists()
    else:
        with fasteners.InterProcessLock(
            interprocess_lock_file(path, dict_lock)
        ):
            return path.exists()


def clean_directory(path: Path, exclude_dir: list = None,
                    exclude_file: list = None, dict_lock: Path = None) -> bool:
    """Remove all files in the directory recursively.

    Args:
        path (Path): Path for the directory
        exclude_dir (list): Exclude directories. All files in directories, are
            not removed.
        exclude_file (list): Exclude files not to be removed.
        dict_lock (Path): A directory saving lock files.

    Returns:
        None
    """
    path = to_path(path)

    if not path.is_dir():
        return False

    if exclude_dir is None:
        exclude_dir = []

    if exclude_file is None:
        exclude_file = []

    if dict_lock is None:
        for p in path.glob('**/*'):
            if p.is_file():
                if True not in [p in d.parts for d in
                                exclude_dir + exclude_file]:
                    p.unlink()
    else:
        with fasteners.InterProcessLock(
            interprocess_lock_file(path, dict_lock)
        ):
            for p in path.glob('**/*'):
                if p.is_file():
                    if True not in [p in d.parts for d in
                                    exclude_dir + exclude_file]:
                        p.unlink()


def copy_directory(from_directory: Path, to_directory: Path,
                   dict_lock: Path = None) -> None:
    """Copy a directory.

    Args:
        from_directory (Path): A copied directory.
        to_directory (Path): A directory located copied directory.
        dict_lock (Path): The path to store lock files.

    Returns:
        None
    """
    if dict_lock is None:
        if not to_directory.is_dir():
            shutil.copytree(from_directory, to_directory)
    else:
        with fasteners.InterProcessLock(
                interprocess_lock_file(from_directory, dict_lock)):
            with fasteners.InterProcessLock(
                    interprocess_lock_file(to_directory, dict_lock)):
                if not to_directory.is_dir():
                    shutil.copytree(from_directory, to_directory)


def create_yaml(path: Path, content: dict, dict_lock: Path = None) -> None:
    """Create a yaml file.

    Args:
        path (Path): The path of the created yaml file.
        content (dict): The content of the created yaml file.
        dict_lock (Path): The path to store lock files.

    Returns:
        None
    """
    if dict_lock is None:
        with open(path, 'w') as f:
            f.write(yaml.dump(content, default_flow_style=False))
    else:
        with fasteners.InterProcessLock(
                interprocess_lock_file(path, dict_lock)):
            with open(path, 'w') as f:
                f.write(yaml.dump(content, default_flow_style=False))


def file_create(path: Path, content: str, dict_lock: Path = None) -> None:
    """Create a text file.

    Args:
        path (Path): The path of the created file.
        content (str): The content of the created file.
        dict_lock (Path): The path to store lock files.

    Returns:
        None
    """
    if dict_lock is None:
        with open(path, 'w') as f:
            f.write(content)
    else:
        with fasteners.InterProcessLock(
                interprocess_lock_file(path, dict_lock)):
            with open(path, 'w') as f:
                f.write(content)


def file_delete(path: Path, dict_lock: Path = None) -> None:
    """Delete a file.

    Args:
        path (Path): A deleted file path.
        dict_lock (Path): A path to store lock files.

    Returns:
        None
    """
    if path.exists():
        if dict_lock is None:
            path.unlink()
        else:
            with fasteners.InterProcessLock(
                    interprocess_lock_file(path, dict_lock)):
                path.unlink()


def file_read(path: Path, dict_lock: Path = None) -> str:
    """Read a file.

    Args:
        path (Path): A path of reading file.
        dict_lock (Path): A path to store lock files.

    Returns:
        str: A content of read file.
    """
    lines = None
    if path.exists():
        if dict_lock is None:
            with open(path, 'r') as f:
                lines = f.read()
        else:
            with fasteners.InterProcessLock(
                    interprocess_lock_file(path, dict_lock)):
                with open(path, 'r') as f:
                    lines = f.read()

    return lines


def get_basename(path: Path):
    """Get a base name of a path without an extension.

    Args:
        path (Path): A path to get a base name.

    Returns:
        str: A base name without an extension.
    """
    return path.stem


def get_dict_files(directory: Path, pattern: str, dict_lock: Path = None) ->\
        List[Path]:
    """Get files matching a pattern in a directory.

    Args:
        directory (Path): A directory to search files.
        pattern (str): A regular expression.
        dict_lock (Path): A directory to store lock files.

    Returns:
        list: Matched files.
    """
    if directory.exists():
        if dict_lock is None:
            files = list(directory.glob(pattern))
            if len(files) > 0:
                files.sort()
            return files
        else:
            with fasteners.InterProcessLock(
                    interprocess_lock_file(directory, dict_lock)):
                files = list(directory.glob(pattern))
                if len(files) > 0:
                    files.sort()
                return files


def get_file_hp_ready(path: Path, dict_lock: Path = None) -> list:
    """Get files in hp/ready directory.

    Args:
        path (Path): A path to hp/ready directory.
        dict_lock (Path): A directory to store lock files.

    Returns:
        list: Files in hp/ready directory.
    """

    return get_dict_files(
        path / aiaccel.dict_hp_ready,
        '*.{}'.format(aiaccel.extension_hp),
        dict_lock=dict_lock
    )


def get_file_hp_running(path, dict_lock=None):
    """Get files in hp/running directory.

    Args:
        path (Path): A path to hp/running directory.
        dict_lock (Path): A directory to store lock files.

    Returns:
        list: Files in hp/running directory.
    """

    return get_dict_files(
        path / aiaccel.dict_hp_running,
        '*.{}'.format(aiaccel.extension_hp),
        dict_lock=dict_lock
    )


def get_file_hp_finished(path, dict_lock=None):
    """Get files in hp/finished directory.

    Args:
        path (Path): A path to hp/finished directory.
        dict_lock (Path): A directory to store lock files.

    Returns:
        list: Files in hp/finished directory.
    """

    return get_dict_files(
        path / aiaccel.dict_hp_finished,
        '*.{}'.format(aiaccel.extension_hp),
        dict_lock=dict_lock
    )


def get_file_result(path, dict_lock=None):
    """Get files in result directory.

    Args:
        path (Path): A path to result directory.
        dict_lock (Path): A directory to store lock files.

    Returns:
        list: Files in result directory.
    """

    return get_dict_files(
        path / aiaccel.dict_result,
        '*.{}'.format(aiaccel.extension_result),
        dict_lock=dict_lock
    )


def interprocess_lock_file(path: Path, dict_lock: Path) -> Path:
    """Get a directory of storing lock files.

    Args:
        path (Path): This base name directory will be created in a
            dict_lock directory.
        dict_lock (Path): A directory to store lock files.

    Returns:
        Path: A directory which path and dict_lock is joined.
    """
    return dict_lock / path.parent.name


def load_yaml(path: Path, dict_lock: Path = None) -> dict:
    """Load a content of a yaml file.

    Args:
        path (Path): A path of a yaml file.
        dict_lock (Path): A directory to store lock files.

    Returns:
        dict: A loaded content.
    """
    if dict_lock is None:
        with open(path, 'r') as f:
            yml = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        with fasteners.InterProcessLock(
                interprocess_lock_file(path, dict_lock)):
            with open(path, 'r') as f:
                yml = yaml.load(f, Loader=yaml.SafeLoader)

    return yml


def make_directory(d: Path, dict_lock: Path = None) -> None:
    """Make a directory.

    Args:
        d (Path): A path of making directory.
        dict_lock (Path): A directory to store lock files.

    Returns:
        None
    """
    if dict_lock is None:
        if not d.exists():
            d.mkdir()
    else:
        with fasteners.InterProcessLock(
                interprocess_lock_file(d, dict_lock)):
            if not d.exists():
                d.mkdir()


def make_directories(ds: list, dict_lock: Path = None) -> None:
    """Make directories.

    Args:
        ds (List[Path]): A list of making directories.
        dict_lock (Path): A directory to store lock files.

    Returns:
        None
    """
    for d in ds:
        if dict_lock is None:
            if not d.is_dir() and d.exists():
                d.unlink()
            make_directory(d)
        else:
            with fasteners.InterProcessLock(
                    interprocess_lock_file(d, dict_lock)):
                if not d.is_dir() and d.exists():
                    d.unlink()
                make_directory(d)


def move_file(from_file: Path, to_file: Path, dict_lock: Path = None) -> None:
    """Move a file.

    Args:
        from_file (Path): A moved path file from.
        to_file (Path): A moved path file to.
        dict_lock (Path): A directory to store lock files.

    Returns:
        None
    """
    content = load_yaml(from_file, dict_lock)
    file_delete(from_file, dict_lock)
    create_yaml(to_file, content, dict_lock)


def to_path(path):
    """Convert to Path object.

    Args:
        path (Any): An any path object.

    Returns:
        Path: A converted path object.
    """
    if isinstance(path, Path):
        return path.resolve()

    return Path(path).resolve()
