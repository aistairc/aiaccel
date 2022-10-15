from __future__ import annotations

import re
import subprocess
import threading
from subprocess import Popen
from typing import TYPE_CHECKING, List, Union

import psutil

if TYPE_CHECKING:
    from aiaccel.master.abci_master import AbciMaster
    from aiaccel.master.abstract_master import AbstractMaster
    from aiaccel.master.local_master import LocalMaster

import datetime


def exec_runner(command: list, silent: bool = True) -> Popen:
    """Execute a subprocess with command.

    Args:
        command (list): A command list
        silent (bool): An option for silent

    Returns:
        Popen: An opened process object.
    """
    # if silent:
    #     return subprocess.Popen(command, stdout=subprocess.DEVNULL,
    #                             stderr=subprocess.DEVNULL)
    # else:
    #     return subprocess.Popen(command, stdout=subprocess.PIPE,
    #                             stderr=subprocess.STDOUT)
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )


def subprocess_ps() -> List[dict]:
    """Get a ps result as a list.

    Returns:
        List[dict]: A ps result.
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


def parse_psaux(outputs: List[bytes]) -> List[dict]:
    """Parse a list of bytes which got from subprocess.

    Args:
        outputs (List[bytes]): A result of ps command using subprocess.

    Returns:
        List[dict]: A parsed result of a ps command.
    """
    output = [o.decode('utf-8') for o in outputs if len(o) > 0]
    headers = [h for h in ' '.join(output[0].strip().split()).split() if h]
    indexes = [output[0].strip().find(h) for h in headers]
    split_indexes = [
        [indexes[i], indexes[i + 1]] if i < len(indexes) - 1
        else [indexes[i], -1] for i in range(0, len(indexes))
    ]
    raw_data = map(
        lambda s: [s[i[0]:i[1]].strip() for i in split_indexes],
        output[1:]
    )

    return [dict(zip(headers, r)) for r in raw_data]


'''
def ps2joblist() -> List[dict]:
    """Get a ps result and convert to a job list format.

    Returns:
        List[dict]: A job list of ps result.

    Raises:
        KeyError: Causes when required keys are not contained in a ps result.
    """
    output = subprocess.Popen(
        ['/bin/ps', '-eo', 'pid,user,stat,lstart,args'],
        stdout=subprocess.PIPE
    ).stdout.readlines()
    output_dict = parse_psaux(output)
    job_list = []
    for o in output_dict:
        d = {
            'job-ID': o['PID'], 'prior': None, 'user': o['USER'],
            'state': o['STAT'], 'queue': None, 'jclass': None,
            'slots': None, 'ja-task-ID': None
        }

        if 'COMMAND' in o:
            d['name'] = o['COMMAND']
        elif 'ARGS' in o:
            d['name'] = o['ARGS']
        else:
            raise KeyError

        if 'START' in o:
            d['submit/start at'] = o['START']
        elif 'STARTED' in o:
            d['submit/start at'] = o['STARTED']
        else:
            raise KeyError

        job_list.append(d)

    return job_list
'''


def ps2joblist() -> List[dict]:
    """Get a ps result and convert to a job list format.

    Returns:
        List[dict]: A job list of ps result.

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
            'slots': None, 'ja-task-ID': None, 'name': " ".join(p_info.info['cmdline']),
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

    Attributes:
        _parent (Union[AbciMaster, AbstractMaster, LocalMaster]): A reference
            for the caller object.
        _proc (Popen): A reference for subprocess.Popen.
        _module_name (str): A module name which the subprocess is attached.
            For example, 'Optimizer'.
        _sleep_time (int): A sleep time each loop.
    """

    def __init__(
        self,
        parent: Union[AbciMaster, AbstractMaster, LocalMaster],
        proc: subprocess.Popen,
        module_name: str,
        # resource_name: str,
        # storage: Storage
    ) -> None:
        """Initial method for OutputHandler.

        Args:
            parent (Union[AbciMaster, AbstractMaster, LocalMaster]): A
                reference for the caller object.
            proc (Popen): A reference for subprocess.Popen.
            module_name (str): A module name which the subprocess is attached.
                For example, 'Optimizer'.
        """
        super(OutputHandler, self).__init__()
        self._parent = parent
        self._proc = proc
        self._module_name = module_name
        self._sleep_time = 1
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        """Main thread.

        Returns:
            None
        """
        while True:
            if self._proc.stdout is None:
                break

            line = self._proc.stdout.readline().decode().strip()

            if line:
                print(line, flush=True)

            if not line and self._proc.poll() is not None:
                self._parent.logger.debug(f'{self._module_name} process finished.')
                o, e = self._proc.communicate()

                if o:
                    print(o.decode().strip(), flush=True)

                if e:
                    print(e.decode().strip(), flush=True)
                break

            if self._abort:
                break


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
