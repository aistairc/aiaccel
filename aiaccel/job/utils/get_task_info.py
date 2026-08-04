# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import os


def is_array_job() -> bool:
    """Return whether the current process is running as an array job."""
    return "TASK_INDEX" in os.environ and "TASK_STEPSIZE" in os.environ


def get_task_index() -> int:
    """Return ``TASK_INDEX`` as an integer."""
    return int(os.environ["TASK_INDEX"])


def get_task_stepsize() -> int:
    """Return ``TASK_STEPSIZE`` as an integer."""
    return int(os.environ["TASK_STEPSIZE"])


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
