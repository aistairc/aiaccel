# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import os


def split_tasks(task_list: list[Any]) -> list[Any]:
    """
    Return the task shard assigned to the current array job.

    This function uses ``TASK_INDEX`` and ``TASK_STEPSIZE`` from the environment to
    slice ``task_list``. The start position is computed as ``TASK_INDEX - 1``.
    If ``TASK_INDEX`` is not defined, an empty list is returned.

    Args:
        task_list (list[Any]): Full list of tasks to be split across array jobs.

    Returns:
        list[Any]: Tasks assigned to the current array job.
    """
    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        return task_list[start:end]
    else:
        return []
