from aiaccel.optimizer.abstract import AbstractOptimizer
from tests.base_test import BaseTest
import asyncio
import pytest
import shutil
import time
import os
import sys
from unittest.mock import patch


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
        params = [{
            'parameters': [
                {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1},
                {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}
            ]
        }]

        assert self.optimizer.register_new_parameters(params) is None

    def test_register_ready(self, data_dir, work_dir):
        param = {
            'parameters': [
                {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1},
                {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}
            ]
        }
        assert type(self.optimizer.register_ready(param)) is int

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

    def test_inner_loop_pre_process(self, setup_hp_finished):
        assert not self.optimizer.inner_loop_pre_process()

        self.optimizer.pre_process()
        assert self.optimizer.inner_loop_pre_process()

        #setup_hp_finished(len(self.optimizer.params.get_hyperparameters()))
        setup_hp_finished(len(self.optimizer.params.get_parameter_list()))
        assert not self.optimizer.inner_loop_pre_process()

    def test_inner_loop_main_process(self):
        def f(number=1):
            return

        self.optimizer.generate_parameter = f
        assert self.optimizer.inner_loop_main_process()

    def test_inner_loop_post_process(self):
        assert self.optimizer.inner_loop_post_process()
