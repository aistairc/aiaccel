import sys
from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.util.aiaccel import CommandLineArgs
from tests.base_test import BaseTest


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
            assert args.get_xs_from_args() == {
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
