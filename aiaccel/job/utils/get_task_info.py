# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import os


def is_array_job() -> bool:
    """Return whether the current process is running as an array job."""
    return "TASK_INDEX" in os.environ and "TASK_STEPSIZE" in os.environ


def get_task_index() -> int | None:
    """Return the task index specified by the ``TASK_INDEX`` environment variable.

    Returns:
        The value of ``TASK_INDEX``, or ``None`` if the variable is not set.
    """
    index = os.environ.get("TASK_INDEX")
    return int(index) if index is not None else index


def get_task_stepsize() -> int | None:
    """Return the task step size specified by the ``TASK_STEPSIZE`` environment variable.

    Returns:
        The value of ``TASK_STEPSIZE``, or ``None`` if the variable is not set.
    """
    task_stepsize = os.environ.get("TASK_STEPSIZE")
    return int(task_stepsize) if task_stepsize is not None else None


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
    if (task_index := get_task_index()) is not None and (task_stepsize := get_task_stepsize()) is not None:
        start = task_index - 1
        end = start + task_stepsize

        return task_list[start:end]
    else:
        return task_list
