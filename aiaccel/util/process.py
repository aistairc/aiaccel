from __future__ import annotations

import copy
import re
import subprocess
import threading
from subprocess import Popen
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:  # pragma: no cover
    from aiaccel.master.abci_master import AbciMaster
    from aiaccel.master.abstract_master import AbstractMaster
    from aiaccel.master.local_master import LocalMaster
    from aiaccel.storage.storage import Storage

import datetime

from aiaccel.util.time_tools import (get_time_now_object,
                                     get_time_string_from_object)


def exec_runner(command: list) -> Popen:
    """Execute a subprocess with command.

    Args:
        command (list): A command list

    Returns:
        Popen: An opened process object.
    """
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def subprocess_ps() -> list[dict]:
    """Get a ps result as a list.

    Returns:
        list[dict]: A ps result.
    """
    commands = ['ps', 'xu']
    res = subprocess.run(commands, stdout=subprocess.PIPE)
    res = res.stdout.decode('utf-8')
    stats = res.split('\n')
    stats_zero = re.split(' +', stats[0])
    stats_zero = [s for s in stats_zero if s != '']
    pid_order = stats_zero.index('PID')
    command_order = stats_zero.index('COMMAND')
    ret = []

    for s in range(1, len(stats)):
        pstat = re.split(' +', stats[s])
        pstat = [s for s in pstat if s != '']

        if len(pstat) < command_order - 1:
            continue

        ret.append({
            'PID': pstat[pid_order],
            'COMMAND': pstat[command_order],
            'full': stats[s]})

    return ret


def ps2joblist() -> list[dict]:
    """Get a ps result and convert to a job list format.

    Returns:
        list[dict]: A job list of ps result.

    Raises:
        KeyError: Causes when required keys are not contained in a ps result.
    """

    job_list = []

    for p_info in psutil.process_iter(['pid', 'username', 'status', 'create_time', 'cmdline']):
        # p_info = proc.as_dict(
        #    attrs=['pid', 'username', 'status', 'create_time', 'cmdline'])
        d = {
            'job-ID': p_info.info['pid'], 'prior': None, 'user': p_info.info['username'],
            'state': p_info.info['status'], 'queue': None, 'jclass': None,
            'slots': None, 'ja-task-ID': None, 'name': " ".join(p_info.info['cmdline'] or []),
            'submit/start at': datetime.datetime.fromtimestamp(
                p_info.info['create_time']).strftime("%Y-%m-%d %H:%M:%S")
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
    args = ['/bin/kill', f'{pid}']
    subprocess.Popen(args, stdout=subprocess.PIPE)


class OutputHandler(threading.Thread):
    """A class to print subprocess outputs.

    Args:
        parent (AbciMaster | AbstractMaster | LocalMaster): A
            reference for the caller object.
        proc (Popen): A reference for subprocess.Popen.
        module_name (str): A module name which the subprocess is attached.
            For example, 'Optimizer'.
        trial_id (int): Trial id.
        storage (Storage | None, optional): Storage object. Defaults to
            None.

    Attributes:
        _parent (AbciMaster | AbstractMaster | LocalMaster): A reference
            for the caller object.
        _proc (Popen): A reference for subprocess.Popen.
        _module_name (str): A module name which the subprocess is attached.
            For example, 'Optimizer'.
        _sleep_time (int): A sleep time each loop.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        super(OutputHandler, self).__init__()
        self._proc = proc
        self._sleep_time = 1
        self._abort = False

        self._returncode = None
        self._stdouts = []
        self._stderrs = []
        self._start_time = None
        self._end_time = None

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        """Main thread.

        Returns:
            None
        """
        self._start_time = get_time_now_object()
        self._returncode = None
        self._stdouts = []
        self._stderrs = []

        while True:
            if self._proc.stdout is None:
                break

            stdout = self._proc.stdout.readline().decode().strip()
            stderr = self._proc.stderr.readline().decode().strip()

            if stdout:
                self._stdouts.append(stdout)

            if stderr:
                self._stderrs.append(stderr)

            returncode = self._proc.poll()
            if not (stdout or stderr) and returncode is not None:
                break

            if self._abort:
                break
    
        self._returncode = returncode
        self._end_time = get_time_now_object()

    def get_stdouts(self) -> list[str]:
        return copy.deepcopy(self._stdouts)

    def get_stderrs(self) -> list[str]:
        return copy.deepcopy(self._stderrs)
    
    def get_start_time(self) -> str:
        return get_time_string_from_object(self._start_time)
    
    def get_end_time(self) -> str:
        return get_time_string_from_object(self._end_time)
    
    def get_returncode(self):
        return self._returncode


def is_process_running(pid: int) -> bool:
    """Check the process is running or not.

    Args:
        pid (int): A pid.

    Returns:
        bool: The process is running or not.
    """
    status = [
        "running",
        "sleeping",
        "disk-sleep",
        "stopped",
        "tracing-stop",
        "waking",
        "idle"
    ]

    try:
        p = psutil.Process(pid)
        return p.status() in status
    except psutil.NoSuchProcess:
        return False
