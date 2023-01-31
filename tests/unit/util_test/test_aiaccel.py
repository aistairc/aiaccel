import sys
from unittest.mock import patch

import numpy as np
from aiaccel.util.aiaccel import Messages, Run, WrapperInterface, CommandLineArgs

from tests.base_test import BaseTest
import pytest


def test_create_message():
    msg = Messages("test")
    msg.create_message("test", "hoge")
    assert msg.get("test") == "test:hoge"

    msg.create_message("test", "")
    assert msg.get("test") == "test:"

    msg.create_message("test", 123)
    assert msg.get("test") == "test:123"

    msg.create_message("test", [1, 2, 3])
    assert msg.get("test") == "test:1@2@3"


def test_out():
    msg = Messages("test")
    msg.create_message("test", [1, 2, 3])
    msg.create_message("test", [1, 2, 3])
    msg.create_message("test", [1, 2, 3])
    assert msg.d["test"].out(all=True) is None
    assert msg.d["test"].out(all=False) is None

def test_parse_result():
    msg = Messages("test")
    assert msg.parse("test", "test:1") == ["1"]
    assert msg.parse("test", "aa\ntest:123\nbbbbb") == ["123"]
    assert msg.parse("test", "test1:1") == [""]
    assert msg.parse("test", "test:1@2@3") == ["1", "2", "3"]
    assert msg.parse("test", "test:1@2@3@[4,5,6]") == ["1", "2", "3", "[4,5,6]"]
    assert msg.parse("test", "test:1@2@3@(4,5,6)") == ["1", "2", "3", "(4,5,6)"]
    assert msg.parse("test", "") == [""]
    assert msg.parse("test", "test:None") == ["None"]


def main(p):

    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"], p["x6"], p["x7"], p["x8"], p["x9"], p["x10"]])
    y = np.sum(x ** 2)

    return float(y)

def invalid_func(p):
    return [1, 2, 3]

#
# Run test
#
class TestRun(BaseTest):

    def get_test_args(self):
        return [
            "wapper.py",
            f"--config={self.config_json}",
            "--trial_id=0001",
            "--x1=0.1",
            "--x2=0.2",
            "--x3=0.3",
            "--x4=0.4",
            "--x5=0.5",
            "--x6=0.6",
            "--x7=0.7",
            "--x8=0.8",
            "--x9=0.9",
            "--x10=1.0"
            # "--i=0.0001"
        ]

    @property
    def test_hp(self):
        return [
            {
                "name": "x1",
                "type": "uniform_float",
                "step": 1.0,
                "log": False,
                "base": 10,
                "lower": 0,
                "upper": 5
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "step": 1.0,
                "log": False,
                "base": 10,
                "lower": 0,
                "upper": 5
            }
        ]

#
# Wrapper Interface
#
def wrapper_interface():
    wrp = WrapperInterface()
    assert wrp.get_data('objective_y: objective_err:') == (None, None)
    assert wrp.get_data('objective_y:123 objective_err:err') == ('123', 'err')


class TestCommandLineArgs(BaseTest):

    def test_command_line_args(self):
        commandline_args = [
                "wapper.py",
                "--trial_id","1",
                "--x1", "0.1",
                "--x2", "0.2",
                "--x3", "0.3",
                "--x4", "0.4",
                "--x5", "0.5",
                "--x6", "0.6",
                "--x7", "0.7",
                "--x8", "0.8",
                "--x9", "0.9",
                "--x10", "1.0"
        ]

        with patch.object(sys, 'argv', commandline_args):
            args = CommandLineArgs()

            assert args.args.trial_id == 1
            assert args.args.x1 == 0.1
            assert args.args.x2 == 0.2
            assert args.args.x3 == 0.3
            assert args.args.x4 == 0.4
            assert args.args.x5 == 0.5
            assert args.args.x6 == 0.6
            assert args.args.x7 == 0.7
            assert args.args.x8 == 0.8
            assert args.args.x9 == 0.9
            assert args.args.x10 == 1.0

            with pytest.raises(AttributeError):
                args.args.i


    def test_get_xs(self):
        commandline_args = [
                "wapper.py",
                "--trial_id","1",
                "--x1", "0.1",
                "--x2", "0.2",
                "--x3", "0.3",
                "--x4", "0.4",
                "--x5", "0.5",
                "--x6", "0.6",
                "--x7", "0.7",
                "--x8", "0.8",
                "--x9", "0.9",
                "--x10", "1.0"
        ]

        with patch.object(sys, 'argv', commandline_args):
            args = CommandLineArgs()
            assert args.get_xs() == {
                "x1": 0.1,
                "x2": 0.2,
                "x3": 0.3,
                "x4": 0.4,
                "x5": 0.5,
                "x6": 0.6,
                "x7": 0.7,
                "x8": 0.8,
                "x9": 0.9,
                "x10": 1.0
            }
