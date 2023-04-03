import asyncio
import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.common import (data_type_categorical, data_type_ordinal,
                            data_type_uniform_float, data_type_uniform_int)
from aiaccel.optimizer import AbstractOptimizer
from tests.base_test import BaseTest


async def async_function(func):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, func)


async def make_directory(sleep_time, d):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    os.mkdir(d)


class TestAbstractOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir):
        options = {
            'config': self.config_json,
            'resume': 0,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        self.optimizer = AbstractOptimizer(options)
        yield
        self.optimizer = None

    def test_all_parameters_processed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'hp_ready', 0)
            m.setattr(self.optimizer, 'hp_running', 0)
            m.setattr(self.optimizer, 'all_parameters_generated', True)
            assert self.optimizer.all_parameters_processed()

    def test_all_parameters_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'trial_number', 0)
            m.setattr(self.optimizer, 'hp_finished', 0)
            m.setattr(self.optimizer, 'hp_ready', 0)
            m.setattr(self.optimizer, 'hp_running', 0)
            assert self.optimizer.all_parameters_registered()

    def test_register_new_parameters(self):
        params = [
            {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1},
            {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}
        ]

        assert self.optimizer.register_new_parameters(params) is None

    def test_generate_initial_parameter(self):
        with patch.object(self.optimizer.params, 'sample', return_value=[]):
            assert self.optimizer.generate_initial_parameter() == []

        p = [
            {'name': "x1", 'type': 'FLOAT', 'value': 1.0},
            {'name': "x2", 'type': 'FLOAT', 'value': 2.0},
        ]

        with patch.object(self.optimizer.params, 'sample', return_value=p):
            assert self.optimizer.generate_initial_parameter() == [
                {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 1.0},
                {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 2.0}
            ]

    def test_generate_parameter(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = self.optimizer.generate_parameter()

    def test_get_pool_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.optimizer.config.num_node, 'get', lambda: 10)
            m.setattr(self.optimizer.storage, 'get_num_running', lambda: 1)
            m.setattr(self.optimizer.storage, 'get_num_ready', lambda: 1)
            assert self.optimizer.get_pool_size() == 10 - 1 - 1

    def test_generate_new_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'num_of_generated_parameter', 0)
            m.setattr(self.optimizer, 'generate_initial_parameter', lambda: None)
            assert self.optimizer.generate_new_parameter() is None

            m.setattr(self.optimizer, 'num_of_generated_parameter', 1)
            m.setattr(self.optimizer, 'generate_parameter', lambda: None)
            assert self.optimizer.generate_new_parameter() is None

    def test_pre_process(self):
        assert self.optimizer.pre_process() is None

    def test_post_process(self):
        self.optimizer.pre_process()
        assert self.optimizer.post_process() is None

    def test_inner_loop_main_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        initial = [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1},
                   {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}]
        param = [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.2},
                 {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.2}]

        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'generate_initial_parameter', lambda: initial)
            m.setattr(self.optimizer, 'generate_parameter', lambda: param)
            m.setattr(self.optimizer, '_serialize', lambda _: None)
            assert self.optimizer.inner_loop_main_process() is True

        with patch.object(self.optimizer, 'check_finished', return_value=True):
            assert self.optimizer.inner_loop_main_process() is False

        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'all_parameters_processed', lambda: True)
            assert self.optimizer.inner_loop_main_process() is False

        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'all_parameters_registered', lambda: True)
            assert self.optimizer.inner_loop_main_process() is True

        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'get_pool_size', lambda: 0)
            assert self.optimizer.inner_loop_main_process() is True

        with monkeypatch.context() as m:
            m.setattr(self.optimizer, 'generate_new_parameter', lambda: param)
            m.setattr(self.optimizer, 'register_new_parameters', lambda _: None)
            m.setattr(self.optimizer.trial_id, 'increment', lambda: None)
            m.setattr(self.optimizer, '_serialize', lambda _: None)
            assert self.optimizer.inner_loop_main_process() is True

            m.setattr(self.optimizer, 'generate_new_parameter', lambda: [])
            assert self.optimizer.inner_loop_main_process() is True

    def test_cast(self):
        org_params = [{'parameter_name': 'x1', 'type': data_type_uniform_int, 'value': 0.1},
                      {'parameter_name': 'x2', 'type': data_type_uniform_int, 'value': 1.5}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == 0
        assert new_params[1]["value"] == 1

        org_params = [{'parameter_name': 'x1', 'type': data_type_uniform_float, 'value': 0.1},
                      {'parameter_name': 'x2', 'type': data_type_uniform_float, 'value': 1.5}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == 0.1
        assert new_params[1]["value"] == 1.5

        org_params = [{'parameter_name': 'x1', 'type': data_type_categorical, 'value': 'a'},
                      {'parameter_name': 'x2', 'type': data_type_categorical, 'value': 'b'}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == 'a'
        assert new_params[1]["value"] == 'b'

        org_params = [{'parameter_name': 'x1', 'type': data_type_ordinal, 'value': [1, 2, 3]},
                      {'parameter_name': 'x2', 'type': data_type_ordinal, 'value': [4, 5, 6]}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == [1, 2, 3]
        assert new_params[1]["value"] == [4, 5, 6]

        org_params = []
        new_params = self.optimizer.cast(org_params)
        assert new_params == []

        org_params = None
        new_params = self.optimizer.cast(org_params)
        assert new_params is None

    def test_check_error(self):
        self.optimizer.storage.error.all_delete()
        assert self.optimizer.check_error() is True

        self.optimizer.storage.error.set_any_trial_error(trial_id=0, error_message="test_error")
        assert self.optimizer.check_error() is False

    def test__serialize(self):
        self.optimizer._rng = np.random.RandomState(0)
        assert self.optimizer._serialize(0) is None

    def test__deserialize(self):
        self.optimizer._rng = np.random.RandomState(0)
        self.optimizer._serialize(1)
        assert self.optimizer._deserialize(1) is None
