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


def from_qsub(status: str) -> JobStatus:
    """
    Converts a status string from the qsub command to a JobStatus enum value.

    Args:
        status (str): The status string from the qsub command.

    Returns:
        JobStatus: The corresponding JobStatus enum value.

    Raises:
        ValueError: If the status string is not recognized.
    """
    match status:
        case "r":
            return JobStatus.RUNNING
        case "qw" | "h" | "t" | "s" | "S" | "T" | "Rq":
            return JobStatus.WAITING
        case "d" | "Rr":
            return JobStatus.RUNNING
        case "E":
            return JobStatus.ERROR
        case _:
            raise ValueError(f"Unexpected status: {status}")
