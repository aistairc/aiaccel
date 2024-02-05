import os
import subprocess
from subprocess import PIPE

from aiaccel.util import OutputHandler


def test_OutputHandler():
    _ouputhandler = OutputHandler(subprocess.Popen(["ls"], stdout=PIPE, stderr=PIPE))

    _ouputhandler._abort = False

    assert _ouputhandler.abort() is None
    assert _ouputhandler.run() is None

    _ouputhandler._abort = False
    assert _ouputhandler.run() is None

    try:
        _ouputhandler = OutputHandler(subprocess.Popen(["ls"], stdout=None))
        _ouputhandler.run()
        assert False
    except TypeError:
        assert True
