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
from argparse import ArgumentParser
from aiaccel.storage.storage import Storage
# from aiaccel.util.opt import Wrapper
from aiaccel.util.aiaccel import Messages, Run, WrapperInterface, report
from aiaccel.util.aiaccel import Abci
from aiaccel.util.aiaccel import Local
from aiaccel.util.aiaccel import Abstruct

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

    def test_init(self):
        test_argv =["wapper.py", f"--config={str(self.config_json)}", "--trial_id=1"]
        with patch.object(sys, 'argv', test_argv):
            print(sys.argv)
            run = Run()
            assert type(run) == Local

        test_argv =["wapper.py", f"--config={str(self.config_abci)}", "--trial_id=1"]
        with patch.object(sys, 'argv', test_argv):
            run = Run()
            assert type(run) == Abci


class TestAbstruct(BaseTest):
    def test_cast_y(self):
        test_argv =["wapper.py", f"--config={str(self.config_json)}", "--trial_id=1"]
        with patch.object(sys, 'argv', test_argv):
            parser = ArgumentParser()
            parser.add_argument('--config', type=str)
            parser.add_argument('--trial_id', type=str, required=False)
            parser.add_argument('--resume', type=int, default=None)
            parser.add_argument('--clean', nargs='?', const=True, default=False)
            args = parser.parse_known_args()[0]
            run = Abstruct(args)

            assert run.cast_y(1.23, None) == 1.23
            assert run.cast_y(1.23, 'float') == 1.23
            assert run.cast_y(1.23, 'int') == 1
            assert run.cast_y(1.23, 'str') == '1.23'
        
            assert run.cast_y(42, None) == 42
            assert run.cast_y(42, 'float') == 42.0
            assert run.cast_y(42, 'int') == 42
            assert run.cast_y(42, 'str') == '42'

            assert run.cast_y('255', None) == '255'
            assert run.cast_y('255', 'float') == 255.0
            assert run.cast_y('255', 'int') == 255
            assert run.cast_y('255', 'str') == '255'

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

