"""Utility helpers for aiaccel-job integrations."""

from __future__ import annotations

from typing import TypeVar

from collections.abc import Sequence
import os

T = TypeVar("T")


def slice_tasks(tasks: Sequence[T]) -> Sequence[T]:
    """Return the chunk of tasks assigned to the current array worker.

    ``aiaccel-job`` populates ``TASK_INDEX`` (1-based) and ``TASK_STEPSIZE`` in the
    process environment whenever you run in ``*-array`` mode. This helper reads those
    variables and returns the contiguous subset of *tasks* that belongs to the current
    worker. If ``TASK_INDEX`` is absent, the original sequence is returned untouched.

    Parameters
    ----------
    tasks:
        Ordered collection of work items (file paths, configs, etc.).

    Returns
    -------
    Sequence
        Subset corresponding to ``TASK_INDEX`` / ``TASK_STEPSIZE``, or the original
        input when array metadata is missing.

    Raises
    ------
    ValueError
        If the environment variables are malformed.
    """

    if "TASK_INDEX" not in os.environ:
        return tasks

    try:
        task_index = int(os.environ["TASK_INDEX"])
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError("TASK_INDEX must be an integer") from exc

    try:
        task_stepsize = int(os.environ["TASK_STEPSIZE"])
    except KeyError as exc:
        raise ValueError("TASK_STEPSIZE is required when TASK_INDEX is set") from exc
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError("TASK_STEPSIZE must be an integer") from exc

    if task_index <= 0 or task_stepsize <= 0:
        raise ValueError("TASK_INDEX and TASK_STEPSIZE must be positive integers")

    start = task_index - 1
    end = start + task_stepsize

    return tasks[start:end]
