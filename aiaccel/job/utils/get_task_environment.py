# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import os


def get_task_index() -> str | None:
    return os.environ.get("TASK_INDEX", None)


def get_task_stepsize() -> str | None:
    return os.environ.get("TASK_STEPSIZE", None)
