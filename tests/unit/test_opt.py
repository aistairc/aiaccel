import aiaccel
import argparse
import pathlib
import subprocess
import sys
from unittest import mock
from unittest.mock import patch
# from aiaccel.util.opt import Wrapper
from aiaccel.util.aiaccel import Run
from aiaccel.util.aiaccel import Messages
# from aiaccel.util.opt import create_objective
from tests.base_test import BaseTest
from subprocess import CompletedProcess
import numpy as np

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

#
# Run test
#
class TestRun(BaseTest):

    def get_test_args(self):
        return [
            "wapper.py",
            "--config={}".format(self.config_json),
            "--index=0001",
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

    # test module: hashname
    def test_hashname(self):
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            assert run.hashname == "0001"

    # test module parameters
    def test_parameters(self):
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            assert run.parameters["x1"] == 0.1
            assert run.parameters["x2"] == 0.2
            assert run.parameters["x3"] == 0.3
            assert run.parameters["x4"] == 0.4
            assert run.parameters["x5"] == 0.5
            assert run.parameters["x6"] == 0.6
            assert run.parameters["x7"] == 0.7
            assert run.parameters["x8"] == 0.8
            assert run.parameters["x9"] == 0.9
            assert run.parameters["x10"] == 1.0

    # test module: objective
    def test_objective(self):
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            assert run.objective is None
            run.execute_and_report(main)
            assert run.objective is not None

    # test module: exist_error
    def test_exist_error(self):
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            with patch.object(run, "err", ""):
                assert run.exist_error() is False

            with patch.object(run, "err", "hoge"):
                assert run.exist_error() is True

            with patch.object(run, "err", None):
                assert run.exist_error() is False

    # test module: trial_stop
    def test_trial_stop(self):
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            with patch.object(run, "err", "has any error"):
                assert run.trial_stop() is None

    # test module: set_error
    def test_set_error(self):
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            run.set_error("test error")
            assert run.err == "test error"

    # excute
    def test_excute(self):
        # success
        """
            objective_y:42
            ny_data_type:int
            nobjective_error:
        """
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            y = run.execute(main)
            assert y == 3.85
