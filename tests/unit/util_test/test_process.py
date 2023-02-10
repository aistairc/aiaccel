import logging
import os
import subprocess
from pathlib import Path
from subprocess import PIPE, STDOUT
from unittest.mock import patch

import psutil
import pytest
from aiaccel.master.abstract_master import AbstractMaster
from aiaccel.util.process import (OutputHandler, exec_runner,
                                  is_process_running, kill_process, ps2joblist,
                                  subprocess_ps)


def test_exec_runner():
    assert type(exec_runner(['ps'])) is subprocess.Popen
    assert type(exec_runner(['ps'])) is subprocess.Popen


def test_subprocess_ps():
    ret = subprocess_ps()
    assert type(ret) is list


'''
def test_ps2joblist(fake_process):
    """
    ToDo: missing test some branches.
    Returns:
        None
    """
    fake_process.register_subprocess(
        ['/bin/ps', '-eo', 'pid,user,stat,lstart,args'],
        stdout=[
            "PID ARGS                          USER          STAT STARTED\n"
            "1   python wrapper.py -i sample1  root          Ss   Mon Oct "
            "10 00:00:00 2020\n"
            "2   python wrapper.py -i sample2  root          Ss   Mon Oct "
            "10 00:00:10 2020\n"
        ]
    )
    ret = ps2joblist()
    assert type(ret) is list

    fake_process.register_subprocess(
        ['/bin/ps', '-eo', 'pid,user,stat,lstart,args'],
        stdout=[
            "PID COMMAND                       USER          STAT START\n"
            "1   python wrapper.py -i sample1  root          Ss   Mon Oct "
            "10 00:00:00 2020\n"
            "2   python wrapper.py -i sample2  root          Ss   Mon Oct "
            "10 00:00:10 2020\n"
        ]
    )
    ret = ps2joblist()
    assert type(ret) is list

    fake_process.register_subprocess(
        ['/bin/ps', '-eo', 'pid,user,stat,lstart,args'],
        stdout=[
            "PID COMM                          USER          STAT START\n"
            "1   python wrapper.py -i sample1  root          Ss   Mon Oct "
            "10 00:00:00 2020\n"
            "2   python wrapper.py -i sample2  root          Ss   Mon Oct "
            "10 00:00:10 2020\n"
        ]
    )
    try:
        ps2joblist()
        assert False
    except KeyError:
        assert True

    fake_process.register_subprocess(
        ['/bin/ps', '-eo', 'pid,user,stat,lstart,args'],
        stdout=[
            "PID COMMAND                       USER          STAT TIME\n"
            "1   python wrapper.py -i sample1  root          Ss   Mon Oct "
            "10 00:00:00 2020\n"
            "2   python wrapper.py -i sample2  root          Ss   Mon Oct "
            "10 00:00:10 2020\n"
        ]
    )
    try:
        ps2joblist()
        assert False
    except KeyError:
        assert True
'''


def test_ps2joblist():
    ret = ps2joblist()
    assert type(ret) is list


def test_kill_process():
    proc = subprocess.Popen(['sleep', '5'], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    assert kill_process(proc.pid) is None


def test_is_process_running():
    assert is_process_running(os.getpid()) is True
    assert is_process_running(99999999) is False


def test_OutputHandler():
    class dummy:
        def __init__(self):
            self.logger = logging.getLogger('root.master')

    trial_id = 0
    _ouputhandler = OutputHandler(dummy(), subprocess.Popen('ls', stdout=PIPE), 'test', trial_id)

    _ouputhandler._abort = False

    assert _ouputhandler.abort() is None
    assert _ouputhandler.run() is None

    _ouputhandler._abort = False
    assert _ouputhandler.run() is None


    _ouputhandler = OutputHandler(dummy(), subprocess.Popen('ls', stdout=None), 'test', trial_id)
    assert _ouputhandler.run() is None

    o = b'\xe3\x81\x82'
    e = b'\xe3\x81\x82'
    _ouputhandler = OutputHandler(dummy(), subprocess.Popen('ls', stdout=PIPE, stderr=STDOUT), 'test', trial_id)
    with patch.object(_ouputhandler._proc, 'communicate', return_value=(o, e)):
        assert _ouputhandler.run() is None

    o = b'\xe3\x81\x82'
    e = b'\0'
    _ouputhandler = OutputHandler(dummy(), subprocess.Popen('ls', stdout=PIPE, stderr=PIPE), 'test', trial_id)
    with patch.object(_ouputhandler._proc, 'communicate', return_value=(o, e)):
        assert _ouputhandler.run() is None
