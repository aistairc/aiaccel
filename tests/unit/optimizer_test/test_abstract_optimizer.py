import asyncio
import numpy as np
import os
import shutil
import sys
import time
from unittest.mock import patch

import pytest
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.config import load_config

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
        self.optimizer = AbstractOptimizer(self.configs["config.json"])
        yield
        self.optimizer = None

    def test_register_new_parameters(self):
        params = [
            {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1},
            {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}
        ]

        assert self.optimizer.register_new_parameters(params) is None

    def test_generate_parameter(self):
        try:
            self.optimizer.generate_parameter()
            assert False
        except NotImplementedError:
            assert True

    def test_pre_process(self):
        assert self.optimizer.pre_process() is None

    def test_post_process(self):
        self.optimizer.pre_process()
        assert self.optimizer.post_process() is None

    def test_inner_loop_main_process(self):
        
        def dummy_register_new_parameters(new_params):
            return
        def dummy_increment():
            return
        def dummy_serialize(trial_id):
            return
        
        initial = [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1}, {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}]
        param = [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.2}, {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.2}]
        
        with patch.object(self.optimizer, 'generate_initial_parameter', return_value=initial):
            with patch.object(self.optimizer, 'generate_parameter', return_value=param):
                with patch.object(self.optimizer, '_serialize', return_value=None):
                    assert self.optimizer.inner_loop_main_process() is True

        with patch.object(self.optimizer, 'check_finished', return_value=True):
            assert self.optimizer.inner_loop_main_process() is False

        # if pool_size <= 0 or self.hp_ready >= _max_pool_size
        with patch.object(self.optimizer.config.resource, 'num_node', 1):
            with patch.object(self.optimizer.config.optimize, 'trial_number', 4):
                with patch.object(self.optimizer, 'hp_ready', return_value=2):
                    with patch.object(self.optimizer, 'hp_running', return_value=0):
                        with patch.object(self.optimizer, 'hp_finished', return_value=0):
                            assert self.optimizer.inner_loop_main_process() is True

        with patch.object(self.optimizer, 'num_of_generated_parameter', 1):
            with patch.object(self.optimizer, 'generate_parameter', return_value=None):
                assert self.optimizer.inner_loop_main_process() is True

        with patch.object(self.optimizer, 'get_pool_size', return_value=1):
            with patch.object(self.optimizer, 'num_of_generated_parameter', 1):
                with patch.object(self.optimizer, 'generate_parameter', return_value=param):
                    with patch.object(self.optimizer, 'register_new_parameters', dummy_register_new_parameters):
                        with patch.object(self.optimizer.trial_id, 'increment', dummy_increment):
                            with patch.object(self.optimizer, '_serialize', dummy_serialize):
                                with patch.object(self.optimizer, 'all_parameter_generated', False):
                                    assert self.optimizer.inner_loop_main_process() is True
                                with patch.object(self.optimizer, 'all_parameter_generated', True):
                                    assert self.optimizer.inner_loop_main_process() is False

    def test__serialize(self):
        self.optimizer._rng = np.random.RandomState(0)
        assert self.optimizer._serialize(0) is None

    def test__deserialize(self):
        self.optimizer._rng = np.random.RandomState(0)
        self.optimizer._serialize(1)
        assert self.optimizer._deserialize(1) is None

    def test_cast(self):
        org_params = [{'parameter_name': 'x1', 'type': 'INT', 'value': 0.1}, {'parameter_name': 'x2', 'type': 'INT', 'value': 1.5}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == 0
        assert new_params[1]["value"] == 1

        org_params = [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1}, {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 1.5}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == 0.1
        assert new_params[1]["value"] == 1.5

        org_params = [{'parameter_name': 'x1', 'type': 'CATEGORICAL', 'value': 'a'}, {'parameter_name': 'x2', 'type': 'CATEGORICAL', 'value': 'b'}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == 'a'
        assert new_params[1]["value"] == 'b'

        org_params = [{'parameter_name': 'x1', 'type': 'ORDINAL', 'value': [1, 2, 3]}, {'parameter_name': 'x2', 'type': 'ORDINAL', 'value': [4, 5, 6]}]
        new_params = self.optimizer.cast(org_params)
        assert new_params[0]["value"] == [1, 2, 3]
        assert new_params[1]["value"] == [4, 5, 6]

        org_params = []
        new_params = self.optimizer.cast(org_params)
        assert new_params == []

        org_params = None
        new_params = self.optimizer.cast(org_params)
        assert new_params == None

    def test_check_error(self):
        self.optimizer.storage.error.all_delete()
        assert self.optimizer.check_error() is True

        self.optimizer.storage.error.set_any_trial_error(trial_id=0, error_message="test_error")
        assert self.optimizer.check_error() is False

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
