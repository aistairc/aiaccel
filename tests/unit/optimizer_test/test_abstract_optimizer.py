import asyncio
import os
import shutil
import sys
import time
from unittest.mock import patch

import pytest
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
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
        self.optimizer.storage.alive.init_alive()
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
        initial = [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1}, {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}]
        param = [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.2}, {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.2}]
        
        with patch.object(self.optimizer, 'generate_initial_parameter', return_value=initial):
            with patch.object(self.optimizer, 'generate_parameter', return_value=param):
                with patch.object(self.optimizer, '_serialize', return_value=None):
                    assert self.optimizer.inner_loop_main_process() is True

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
