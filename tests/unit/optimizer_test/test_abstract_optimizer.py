from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from tests.base_test import BaseTest
import aiaccel
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
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'optimizer'
        }
        self.optimizer = AbstractOptimizer(options)
        yield
        self.optimizer = None

    def test_create_parameter_files(self):
        params = [{
            'parameters': [
                {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1},
                {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}
            ]
        }]

        assert self.optimizer.create_parameter_files(params) is None

    def test_create_parameter_file(self, data_dir, work_dir):
        param = {
            'parameters': [
                {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.1},
                {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 0.1}
            ]
        }
        shutil.copy(
            data_dir.joinpath('work', aiaccel.dict_hp_finished, '001.hp'),
            work_dir.joinpath(aiaccel.dict_hp_ready, 'iK2.hp'),
        )
        shutil.copy(
            data_dir.joinpath('work', aiaccel.dict_hp_finished, '001.hp'),
            work_dir.joinpath(aiaccel.dict_hp_running, 'ZWe.hp'),
        )
        shutil.copy(
            data_dir.joinpath('work', aiaccel.dict_hp_finished, '001.hp'),
            work_dir.joinpath(aiaccel.dict_hp_finished, 'qhF.hp'),
        )
        # assert self.optimizer.create_parameter_file(param) in 'WCE'
        # change for wd
        assert self.optimizer.create_parameter_file(param) in '000000'

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

    def test_loop_pre_process(self, work_dir):
        work_dir.joinpath(aiaccel.dict_runner).rmdir()
        loop = asyncio.get_event_loop()
        gather = asyncio.gather(
            async_function(self.optimizer.loop_pre_process),
            make_directory(1, work_dir.joinpath(aiaccel.dict_runner))
        )
        loop.run_until_complete(gather)

        assert self.optimizer.check_work_directory()

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

    def test_serialize(self):
        self.optimizer.serialize_datas = {
            'generated_parameter': None,
            'loop_count': 0
        }
        serialized_dict = self.optimizer._serialize()
        assert 'generated_parameter' in serialized_dict
        assert 'loop_count' in serialized_dict

    def test_deserialize(self):
        self.optimizer.pre_process()
        self.optimizer.serialize_datas = {
            'generated_parameter': None,
            'loop_count': 0
        }
        self.optimizer._serialize()
        serialized_dict = self.optimizer._serialize()
        assert self.optimizer._deserialize(serialized_dict) is None
