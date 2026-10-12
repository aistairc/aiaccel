# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from pathlib import Path, PurePosixPath, PureWindowsPath


def is_absolute_path(value: str | Path) -> bool:
    value = str(value)
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _remove_absolute_path(obj: Any, prefix: str, removed: list[str]) -> Any:
    cleaned: Any

    if isinstance(obj, dict):
        cleaned_dict = {}
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (str, Path)) and is_absolute_path(value):
                removed.append(path)
                continue

            cleaned_dict[key] = _remove_absolute_path(value, path, removed)
        cleaned = type(obj)(cleaned_dict)

    elif isinstance(obj, (list, tuple)):
        cleaned_list: list[Any] = []
        for index, value in enumerate(obj):
            path = f"{prefix}[{index}]"
            if isinstance(value, (str, Path)) and is_absolute_path(value):
                removed.append(path)
                continue

            cleaned_list.append(_remove_absolute_path(value, path, removed))

        cleaned = type(obj)(cleaned_list)

    else:
        return obj

    return cleaned


def remove_absolute_path(obj: Any) -> tuple[Any, list[str]]:
    removed: list[str] = []
    return _remove_absolute_path(obj, "", removed), removed
