from __future__ import annotations

import copy
import datetime
import select
import subprocess
import sys
import threading
from typing import Any

import psutil

from aiaccel.common import datetime_format


def ps2joblist() -> list[dict[str, Any]]:
    """Get a ps result and convert to a job list format.

    Returns:
        list[dict]: A job list of ps result.

    Raises:
        KeyError: Causes when required keys are not contained in a ps result.
    """

    job_list = []

    for p_info in psutil.process_iter(["pid", "username", "status", "create_time", "cmdline"]):
        # p_info = proc.as_dict(
        #    attrs=['pid', 'username', 'status', 'create_time', 'cmdline'])
        d = {
            "job-ID": p_info.info["pid"],
            "prior": None,
            "user": p_info.info["username"],
            "state": p_info.info["status"],
            "queue": None,
            "jclass": None,
            "slots": None,
            "ja-task-ID": None,
            "name": " ".join(p_info.info["cmdline"] or []),
            "submit/start at": datetime.datetime.fromtimestamp(p_info.info["create_time"]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        job_list.append(d)

    return job_list


class OutputHandler(threading.Thread):
    """A class to print subprocess outputs.

    Args:
        proc (Popen): A reference for subprocess.Popen.
            For example, 'Optimizer'.
    Attributes:
        _proc (Popen): A reference for subprocess.Popen.
            For example, 'Optimizer'.
        _sleep_time (int): A sleep time each loop.
    """

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        super(OutputHandler, self).__init__()
        self._proc = proc
        self._sleep_time = 1
        self._abort = False

        self._returncode = None
        self._stdouts: list[str] = []
        self._stderrs: list[str] = []
        self._start_time: datetime.datetime | None = None
        self._end_time: datetime.datetime | None = None

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        self._start_time = datetime.datetime.now()
        while True:
            inputs = [self._proc.stdout, self._proc.stderr]
            readable, _, _ = select.select(inputs, [], [], self._sleep_time)
            for s in readable:
                line = s.readline()
                if s is self._proc.stdout and line:
                    self._stdouts.append(line.decode().strip())
                elif s is self._proc.stderr and line:
                    self._stderrs.append(line.decode().strip())
            if self.get_returncode() is not None:
                # After the process has finished, read the remaining output.
                for stream, storage in [(self._proc.stdout, self._stdouts), (self._proc.stderr, self._stderrs)]:
                    if stream is None:
                        continue
                    for line in stream:
                        storage.append(line.decode().strip())
                break
            if self._abort:
                break
        self._end_time = datetime.datetime.now()
        sys.stdout.flush()
        sys.stderr.flush()

    def get_stdouts(self) -> list[str]:
        return copy.deepcopy(self._stdouts)

    def get_stderrs(self) -> list[str]:
        return copy.deepcopy(self._stderrs)

    def get_start_time(self) -> str | None:
        if self._start_time is None:
            return ""
        return self._start_time.strftime(datetime_format)

    def get_end_time(self) -> str | None:
        if self._end_time is None:
            return ""
        return self._end_time.strftime(datetime_format)

    def get_returncode(self) -> int | None:
        return self._proc.poll()

    def raise_exception_if_error(self) -> None:
        """Raise an exception if an error is detected.

        Returns:
            None
        """
        if self._proc.returncode != 0:
            raise RuntimeError(
                f"An error occurred in the subprocess.\n" f"stdout: {self._stdouts}\n" f"stderr: {self._stderrs}"
            )

    def enforce_kill(self) -> None:
        """Enforce killing the subprocess.

        Returns:
            None
        """
        self._proc.kill()
        raise RuntimeError(
            f"An error occurred in the subprocess.\n" f"stdout: {self._stdouts}\n" f"stderr: {self._stderrs}"
        )
