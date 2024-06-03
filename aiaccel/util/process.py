from __future__ import annotations

import copy
import datetime
import os
import select
import subprocess
import sys
import threading
import time
from typing import IO, List, Union, Any

from aiaccel.common import datetime_format


def select_for_win(
        rlist: list[IO[bytes]],
        timeout: int = 1
) -> tuple[List[Union[IO[bytes], Any]]]:
    """Alternative to select.select() on Windows.

    Args:
        rlist (list): A list of IO objects. It waits until ready for reading.
        timeout (int): An integer specifies a time-out in seconds.

    Returns:
        tuple[list, list]: A tuple consisting a list of readable objects
        and Exceptions.
    """
    start_time = time.time()
    readable, errorlist = [], []

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if timeout is not None and elapsed_time >= timeout:
            break
        for fd_r in rlist:
            try:
                if os.read(fd_r.fileno(), 1):
                    readable.append(fd_r)
            except BlockingIOError:
                pass
            except Exception:
                errorlist.append(fd_r)
        if readable or errorlist:
            break
    return readable, errorlist


class OutputHandler(threading.Thread):
    """A class to print subprocess outputs.

    Args:
        proc (Popen): A reference for subprocess.Popen.
            For example, 'Optimizer'.

    Attributes:
        _proc (Popen): A reference for subprocess.Popen.
            For example, 'Optimizer'.
        _sleep_time (int): A sleep time for reading stdout and stderr.
        _abort (bool): A flag to abort the subprocess.
        _returncode (int): A returncode of the subprocess.
        _stdouts (list[str]): A list of stdout.
        _stderrs (list[str]): A list of stderr.
        _start_time (datetime): A start time of the subprocess.
        _end_time (datetime): An end time of the subprocess.
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
        """Abort the subprocess.

        Args:
            None

        Returns:
            None
        """
        self._abort = True

    def run(self) -> None:
        """Override threading.Thread.run

        Args:
            None

        Returns:
            None
        """
        self._start_time = datetime.datetime.now()
        while True:
            inputs = [self._proc.stdout, self._proc.stderr]
            if sys.platform == "win32":
                readable, _ = select_for_win(inputs, self._sleep_time)
            else:
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
        """Get stdout.

        Args:
            None

        Returns:
            list[str]: A list of stdout.
        """
        return copy.deepcopy(self._stdouts)

    def get_stderrs(self) -> list[str]:
        """Get stderr.

        Args:
            None

        Returns:
            list[str]: A list of stderr.
        """
        return copy.deepcopy(self._stderrs)

    def get_start_time(self) -> str:
        """Get a start time of the subprocess.

        Args:
            None

        Returns:
            str: A start time of the subprocess.
        """
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
