import os
import subprocess
from subprocess import PIPE

from aiaccel.util import OutputHandler, ps2joblist


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


def test_OutputHandler():
    _ouputhandler = OutputHandler(subprocess.Popen('ls', stdout=PIPE))

    _ouputhandler._abort = False

    assert _ouputhandler.abort() is None
    assert _ouputhandler.run() is None

    _ouputhandler._abort = False
    assert _ouputhandler.run() is None

    _ouputhandler = OutputHandler(subprocess.Popen('ls', stdout=None))
    assert _ouputhandler.run() is None
