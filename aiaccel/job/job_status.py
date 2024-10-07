from __future__ import annotations

from enum import IntEnum, auto


class JobStatus(IntEnum):
    """
    Represents the status of a job.

    Attributes:
        UNSUBMITTED: The job has not been submitted.
        WAITING: The job is waiting to be executed.
        RUNNING: The job is currently running.
        FINISHED: The job has finished successfully.
        ERROR: The job encountered an error.

    Methods:
        from_qsub(status: str) -> JobStatus:
            Converts a status string from the qsub command to a JobStatus enum value.

    Raises:
        ValueError: If the status string is not recognized.
    """

    UNSUBMITTED = auto()
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ERROR = auto()
