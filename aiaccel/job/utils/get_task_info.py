# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import os


def is_array_job() -> bool:
    """Return whether the current process is running as an array job."""
    return "TASK_INDEX" in os.environ and "TASK_STEPSIZE" in os.environ


def get_task_index() -> int:
    """Return the task index for the current array job.

    Returns:
        The integer value of the ``TASK_INDEX`` environment variable.

    Raises:
        RuntimeError: If ``TASK_INDEX`` is not set.
        ValueError: If ``TASK_INDEX`` cannot be converted to an integer.
    """
    try:
        value = os.environ["TASK_INDEX"]
    except KeyError as error:
        raise RuntimeError("TASK_INDEX is not set. This process is not running as an array job.") from error

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"TASK_INDEX must be an integer, but got {value!r}.") from error


def get_task_stepsize() -> int:
    """Return the task step size for the current array job.

    Returns:
        The integer value of the ``TASK_STEPSIZE`` environment variable.

    Raises:
        RuntimeError: If ``TASK_STEPSIZE`` is not set.
        ValueError: If ``TASK_STEPSIZE`` cannot be converted to an integer.
    """
    try:
        value = os.environ["TASK_STEPSIZE"]
    except KeyError as error:
        raise RuntimeError("TASK_STEPSIZE is not set. This process is not running as an array job.") from error

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"TASK_STEPSIZE must be an integer, but got {value!r}.") from error


def split_tasks(task_list: list[Any]) -> list[Any]:
    """
    Return the task shard assigned to the current array job.

    This function uses ``TASK_INDEX`` and ``TASK_STEPSIZE`` from the environment to
    slice ``task_list``. The start position is computed as ``TASK_INDEX - 1``.
    If ``TASK_INDEX`` is not defined, the input is returned as is.

    Args:
        task_list (list[Any]): Full list of tasks to be split across array jobs.

    Returns:
        list[Any]: Tasks assigned to the current array job.
    """
    if is_array_job():
        start = get_task_index() - 1
        end = start + get_task_stepsize()

        return task_list[start:end]
    else:
        return task_list
