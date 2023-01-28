import dataclasses
import json
from threading import local

import pytest
from aiaccel.config import (BaseConfig, Config, ConfileWrapper,
                            JsonOrYamlObjectConfig, load_config)

from tests.base_test import BaseTest
from unittest.mock import patch
import sys


class TestBaseConfig(object):

    def test_base_config(self):
        BaseConfig.__abstractmethods__ = set()

        @dataclasses.dataclass
        class BaseConfigChild(BaseConfig):
            pass

        c = BaseConfigChild()
        k = c.get_property('key')
        d = c.to_dict()
        assert k is None
        assert d is None


class TestJsonOrYamlObjectConfig(BaseTest):

    @pytest.fixture(autouse=True)
    def setup(self, config_json):
        with open(config_json, 'r') as f:
            config = json.load(f)
        self.config = JsonOrYamlObjectConfig(config, 'json_object')

    def test_init(self):
        try:
            JsonOrYamlObjectConfig({}, 'invalid_type')
            assert False
        except TypeError:
            assert True

    def test_get_property(self):
        none_config = JsonOrYamlObjectConfig([], 'json_object')
        assert none_config.get_property('key') is None
        none_config = JsonOrYamlObjectConfig({}, 'json_object')
        assert none_config.get_property('key') is None
        assert self.config.get_property('sleep_time') == {
            'master': 1,
            'optimizer': 1,
            'scheduler': 1
        }
        invalid_config = JsonOrYamlObjectConfig({'key': []}, 'json_object')
        assert invalid_config.get_property('key', 'key') is None

    def test_to_dict(self):
        assert type(self.config.to_dict()) == dict


class TestConfileWrapper(BaseTest):

    def test_init(self, config_json):
        try:
            ConfileWrapper(self.config, 'invalid_type')
            assert False
        except TypeError:
            assert True

        with open(config_json) as f:
            json_object = json.load(f)

        json_object_config = ConfileWrapper(json_object, 'json_object')
        assert json_object_config.get('generic', 'project_name') == 'sphere'

    def test_get(self, work_dir):
        # value = self.config.get('generic', 'project_name')
        # assert value == 'sphere'
        # assert self.config.get('invalid_key') is None
        value = self.config.workspace.get()
        assert value == str(work_dir)


def test_load_config(config_json, config_yaml):
    try:
        load_config('not_found.file')
        assert False
    except FileNotFoundError:
        assert True

    json_config = load_config(config_json)
    yaml_config = load_config(config_yaml)
    assert json_config.get('generic', 'project_name') == 'sphere'
    assert yaml_config.get('generic', 'project_name') == 'sphere'

    try:
        load_config('setup.py')
        assert False
    except TypeError:
        assert True


def test_config(config_json):
    config = Config(config_json, warn=False, format_check=False)
    config = Config(config_json, warn=False, format_check=True)
    config = Config(config_json, warn=True, format_check=False)
    config = Config(config_json, warn=True, format_check=True)

    assert config.workspace.get() == "/tmp/work"
    assert config.job_command.get() == "python original_main.py"
    assert config.resource_type.get() == "local"
    assert config.num_node.get() == 4
    assert config.abci_group.get() == "gaa"
    assert config.search_algorithm.get() == 'aiaccel.optimizer.NelderMeadOptimizer'
    assert config.goal.get() == "minimize"
    assert config.trial_number.get() == 10
    assert config.name_length.get() == 6
    assert config.init_fail_count.get() == 100
    assert config.cancel_retry.get() == 3
    assert config.cancel_timeout.get() == 60
    assert config.expire_retry.get() == 3
    assert config.expire_timeout.get() == 60
    assert config.finished_retry.get() == 3
    assert config.finished_timeout.get() == 60
    assert config.job_loop_duration.get() == 0.5
    assert config.job_retry.get() == 2
    assert config.job_timeout.get() == 60
    assert config.kill_retry.get() == 3
    assert config.kill_timeout.get() == 60
    assert config.result_retry.get() == 1
    assert config.runner_retry.get() == 3
    assert config.runner_timeout.get() == 60
    assert config.running_retry.get() == 3
    assert config.running_timeout.get() == 60
    assert config.sleep_time.get() == 0.01
    assert config.master_logfile.get() == "master.log"
    assert config.master_file_log_level.get() == "DEBUG"
    assert config.master_stream_log_level.get() == "DEBUG"
    assert config.optimizer_logfile.get() == "optimizer.log"
    assert config.optimizer_file_log_level.get() == "DEBUG"
    assert config.optimizer_stream_log_level.get() == "DEBUG"
    assert config.scheduler_logfile.get() == "scheduler.log"
    assert config.scheduler_file_log_level.get() == "DEBUG"
    assert config.scheduler_stream_log_level.get() == "DEBUG"
    assert config.hyperparameters.get() == [
        {
            "name": "x1",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [0.74,  1.69,  1.22,  2.09, -3.24, -3.58,  4.13,  2.08]
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [2.98,  2.27,  1.41, -2.10, -3.29, -0.35,  3.87,  4.66,  3.54,  1.17, 3.10]
        },
        {
            "name": "x3",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [3.62,  4.38, -0.40,  2.94,  2.23, -3.07, -2.35, -1.15,  0.89,  2.01, -0.58]
        },
        {
            "name": "x4",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [0.90,  2.00,  4.29, -1.43, -4.02,  2.25,  0.28, -3.00, -0.18,  0.96, -2.09]
        },
        {
            "name": "x5",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [1.99,  3.90,  3.10,  0.06, -3.83,  1.16,  0.93,  0.01, -3.89, -2.04, 0.33]
        },
        {
            "name": "x6",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [-2.78,  4.62, -2.71, -0.66, -2.48, -3.07, -0.04,  0.87,  3.89,  2.68, -4.18]
        },
        {
            "name": "x7",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [1.00, -2.20,  1.82,  0.52, -3.82,  3.23,  3.16,  2.41,  4.99, -2.01, 2.05]
        },
        {
            "name": "x8",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [4.97,  4.77, -0.60, -1.75, -0.68,  2.16,  3.70, -0.86, -2.32, -4.90, -1.4]
        },
        {
            "name": "x9",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [1.98, -3.66, -3.17,  2.95,  2.27, -3.19, -1.18, -1.60,  4.96, -2.39, 1.64]
        },
        {
            "name": "x10",
            "type": "uniform_float",
            "log": False,
            "lower": -5.0,
            "upper": 5.0,
            "initial": [4.03,  3.59, -2.06,  3.03,  3.10, -2.84, -4.57, -0.62, -1.14,  2.15, 1.92]
        }
    ]


def test_config_set(config_json):
    config = Config(config_json, warn=False, format_check=False)
    assert config.workspace.set("aaa") is None
    assert config.workspace.get() == "aaa"
    with pytest.raises(TypeError):
        assert config.workspace.set(123)


def test_empty_if_error(config_json):
    config = Config(config_json, warn=False, format_check=False)
    config.workspace.set("aaa")
    config.workspace.empty_if_error()

    with patch.object(sys, 'exit', return_value=None):
        config.workspace.set("")
        config.workspace.empty_if_error()


def test_value(config_json):
    config = Config(config_json)
    config.workspace.Value == "/tmp/work"


def test_config_not_exists():
    with pytest.raises(FileNotFoundError):
        config = Config("?")
