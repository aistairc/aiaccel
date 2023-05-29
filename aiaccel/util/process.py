from __future__ import annotations

import copy
import datetime
import re
import subprocess
import threading
from subprocess import Popen
from typing import Any

import psutil

from aiaccel.util.time_tools import format_datetime_to_str


def exec_runner(command: list[Any]) -> Popen[bytes]:
    """Execute a subprocess with command.

    Args:
        command (list): A command list

    Returns:
        Popen: An opened process object.
    """
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def subprocess_ps() -> list[dict[str, Any]]:
    """Get a ps result as a list.

    Returns:
        list[dict]: A ps result.
    """
    commands = ["ps", "xu"]
    res = subprocess.run(commands, stdout=subprocess.PIPE)
    message = res.stdout.decode("utf-8")
    stats = message.split("\n")
    stats_zero = re.split(" +", stats[0])
    stats_zero = [s for s in stats_zero if s != ""]
    pid_order = stats_zero.index("PID")
    command_order = stats_zero.index("COMMAND")
    ret = []

    for s in range(1, len(stats)):
        pstat = re.split(" +", stats[s])
        pstat = [s for s in pstat if s != ""]

        if len(pstat) < command_order - 1:
            continue

        ret.append({"PID": pstat[pid_order], "COMMAND": pstat[command_order], "full": stats[s]})

    return ret


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


def kill_process(pid: int) -> None:
    """Kill a process with PID using subprocess.

    Args:
        pid (int): A PID.

    Returns:
        None
    """
    args = ["/bin/kill", f"{pid}"]
    subprocess.Popen(args, stdout=subprocess.PIPE)


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
        """Main thread.

        Returns:
            None
        """
        self._start_time = datetime.datetime.now()
        self._stdouts = []
        self._stderrs = []

        while True:
            if self._proc.stdout is None:
                break

            stdout = self._proc.stdout.readline().decode().strip()
            if stdout:
                self._stdouts.append(stdout)

            if self._proc.stderr is not None:
                stderr = self._proc.stderr.readline().decode().strip()
                if stderr:
                    self._stderrs.append(stderr)
            else:
                stderr = None

            if not (stdout or stderr) and self.get_returncode() is not None:
                break

            if self._abort:
                break

        self._end_time = datetime.datetime.now()

    def get_stdouts(self) -> list[str]:
        return copy.deepcopy(self._stdouts)

    def get_stderrs(self) -> list[str]:
        return copy.deepcopy(self._stderrs)

    def get_start_time(self) -> str | None:
        if self._start_time is None:
            return ""
        return format_datetime_to_str(self._start_time)

    def get_end_time(self) -> str | None:
        if self._end_time is None:
            return ""
        return format_datetime_to_str(self._end_time)

    def get_returncode(self) -> int | None:
        return self._proc.poll()


def is_process_running(pid: int) -> bool:
    """Check the process is running or not.

    Args:
        pid (int): A pid.

    Returns:
        bool: The process is running or not.
    """
    status = ["running", "sleeping", "disk-sleep", "stopped", "tracing-stop", "waking", "idle"]

    try:
        p = psutil.Process(pid)
        return p.status() in status
    except psutil.NoSuchProcess:
        return False
