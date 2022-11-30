import argparse
import pathlib
import subprocess
import sys
from subprocess import CompletedProcess
from textwrap import wrap
from unittest import mock
from unittest.mock import patch

import aiaccel
import numpy as np
from aiaccel.storage.storage import Storage
# from aiaccel.util.opt import Wrapper
from aiaccel.util.aiaccel import Messages, Run, WrapperInterface, report

# from aiaccel.util.opt import create_objective
from tests.base_test import BaseTest


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

    # test module: trial_id
    def test_trial_id(self):
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            assert run.trial_id == "0001"

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

            assert run.execute_and_report(invalid_func) is None

            with patch.object(run, "args", {'trial_id': 1, 'config':None}):
                assert run.execute_and_report(invalid_func) is None

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
            
            with patch.object(run, "err", ""):
                assert run.trial_stop() is None

    # test module: set_error
    def test_set_error(self):
        # for i in range(10):
        #     self.storage.result.set_any_trial_objective(
        #         trial_id=i,
        #         objective=0.0
        #     )
        # self.storage.result.all_delete()
        # self.storage.timestamp.all_delete()
        # self.storage.errors.all_delete()
        with patch.object(sys, 'argv', self.get_test_args()):
            run = Run()
            run.index = '0001'
            run.set_error("test error")
            assert run.err == "test error"
            assert run.error == "test error"

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


#
# Wrapper Interface
#
def wrapper_interface():
    wrp = WrapperInterface()
    assert wrp.get_data('objective_y: objective_err:') == (None, None)
    assert wrp.get_data('objective_y:123 objective_err:err') == ('123', 'err')

def test_report():
    assert report(objective_y=123, objective_err='error') is None
    assert report(objective_y=[123], objective_err='') is None

