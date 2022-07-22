import os
from aiaccel.util.process import exec_runner, subprocess_ps, ps2joblist,\
    kill_process
from aiaccel.util.process import is_process_running
import subprocess


def test_exec_runner():
    assert type(exec_runner(['ps'], True)) is subprocess.Popen
    assert type(exec_runner(['ps'], True)) is subprocess.Popen


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
    assert is_process_running(os.getpid())
