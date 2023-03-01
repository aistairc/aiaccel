from __future__ import annotations

from pathlib import Path
from typing import Any

import fasteners
import yaml

import aiaccel


def create_yaml(path: Path, content: Any, dict_lock: Path | None = None) -> None:
    """Create a yaml file.

    Args:
        path (Path): The path of the created yaml file.
        content (dict): The content of the created yaml file.
        dict_lock (Path | None, optional): The path to store lock files.
            Defaults to None.

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


def file_create(path: Path, content: str, dict_lock: Path | None = None
                ) -> None:
    """Create a text file.

    Args:
        path (Path): The path of the created file.
        content (str): The content of the created file.
        dict_lock (Path | None, optional): The path to store lock files.
            Defaults to None.

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


def file_delete(path: Path, dict_lock: Path | None = None) -> None:
    """Delete a file.

    Args:
        path (Path): A deleted file path.
        dict_lock (Path | None, optional): A path to store lock files.
            Defaults to None.

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


def file_read(path: Path, dict_lock: Path | None = None) -> str | None:
    """Read a file.

    Args:
        path (Path): A path of reading file.
        dict_lock (Path | None, optional): A path to store lock files.
            Defaults to None.

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


def get_dict_files(directory: Path, pattern: str,
                   dict_lock: Path | None = None) -> list[Path] | None:
    """Get files matching a pattern in a directory.

    Args:
        directory (Path): A directory to search files.
        pattern (str): A regular expression.
        dict_lock (Path | None, optional): A directory to store lock files.
            Defaults to None.

    Returns:
        list | None: Matched files.
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
    else:
        return None


def get_file_result(path: Path, dict_lock: Path | None = None
                    ) -> list[Path] | None:
    """Get files in result directory.

    Args:
        path (Path): A path to result directory.
        dict_lock (Path | None, optional): A directory to store lock files.
            Defaults to None.

    Returns:
        list: Files in result directory.
    """

    return get_dict_files(
        path / aiaccel.dict_result,
        f'*.{aiaccel.extension_result}',
        dict_lock=dict_lock
    )


def get_file_result_hp(path: Path, dict_lock: Path | None = None
                       ) -> Any:
    """Get files in result directory.

    Args:
        path (Path): A path to result directory.
        dict_lock (Path | None, optional): A directory to store lock files.
            Defaults to None.

    Returns:
        list: Files in result directory.
    """

    return get_dict_files(
        path / aiaccel.dict_result,
        f'*.{aiaccel.extension_hp}',
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


def load_yaml(path: Path, dict_lock: Path | None = None) -> dict[str, Any]:
    """Load a content of a yaml file.

    Args:
        path (Path): A path of a yaml file.
        dict_lock (Path | None, optional): A directory to store lock files.
            Defaults to None.

    Returns:
        dict: A loaded content.
    """
    if dict_lock is None:
        with open(path, 'r') as f:
            yml = yaml.load(f, Loader=yaml.UnsafeLoader)
    else:
        with fasteners.InterProcessLock(interprocess_lock_file(path, dict_lock)):
            with open(path, 'r') as f:
                yml = yaml.load(f, Loader=yaml.UnsafeLoader)
    return yml


def make_directory(d: Path, dict_lock: Path | None = None) -> None:
    """Make a directory.

    Args:
        d (Path): A path of making directory.
        dict_lock (Path | None, optional): A directory to store lock files.
            Defaluts to None.

    Returns:
        None
    """
    if dict_lock is None:
        if not d.exists():
            d.mkdir()
    else:
        with fasteners.InterProcessLock(interprocess_lock_file(d, dict_lock)):
            if not d.exists():
                d.mkdir()


def make_directories(ds: list[Path], dict_lock: Path | None = None) -> None:
    """Make directories.

    Args:
        ds (list[Path]): A list of making directories.
        dict_lock (Path | None, optional): A directory to store lock files.
            Defaults to None.

    Returns:
        None
    """
    for d in ds:
        if dict_lock is None:
            if not d.is_dir() and d.exists():
                d.unlink()
            make_directory(d)
        else:
            with fasteners.InterProcessLock(interprocess_lock_file(d, dict_lock)):
                if not d.is_dir() and d.exists():
                    d.unlink()
                make_directory(d)
