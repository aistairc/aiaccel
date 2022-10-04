from aiaccel.module import AbstractModule
from aiaccel.optimizer.random.search import RandomOptimizer
from aiaccel.scheduler.local import LocalScheduler
from tests.base_test import BaseTest
import aiaccel
import asyncio
import logging
import pytest
import time


async def async_function(func):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, func)


async def delay_make_directory(sleep_time, d):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    d.mkdir()


def dummy_break():
    import sys
    sys.exit()


class TestAbstractModule(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_module(self, clean_work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'module_name': 'test'
        }

        self.module = AbstractModule(options)
        self.module.storage.alive.init_alive()
        self.module.logger = logging.getLogger(__name__)
        yield
        self.module = None

    def test_get_each_state_count(self):
        assert self.module.get_each_state_count() is None
        assert self.module.hp_ready == 0
        assert self.module.hp_running == 0
        assert self.module.hp_finished == 0

    def test_get_module_type(self, work_dir):
        module_type = self.module.get_module_type()
        assert module_type is None

        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False
        }

        optimizer = RandomOptimizer(options)
        module_type = optimizer.get_module_type()
        assert module_type == aiaccel.module_type_optimizer

        scheduler = LocalScheduler(options)
        module_type = scheduler.get_module_type()
        assert module_type == aiaccel.module_type_scheduler

    def test_check_finished(self, setup_hp_finished):
        assert not self.module.check_finished()

        setup_hp_finished(int(self.module.config.trial_number.get()))

        assert self.module.check_finished()

    def test_print_dict_state(self):
        assert self.module.print_dict_state() is None

    def test_pre_process(self, work_dir):
        try:
            self.module.pre_process()
            assert False
        except NotImplementedError:
            assert True

    def test_post_process(self):
        try:
            self.module.post_process()
            assert False
        except NotImplementedError:
            assert True

    def test_loop_pre_process(self, work_dir):
        try:
            self.module.post_process()
            assert False
        except NotImplementedError:
            assert True

    def test_loop_post_process(self):
        try:
            self.module.loop_post_process()
            assert False
        except NotImplementedError:
            assert True

    def test_inner_loop_pre_process(self):
        try:
            self.module.inner_loop_pre_process()
            assert False
        except NotImplementedError:
            assert True

    def test_inner_loop_main_process(self):
        try:
            self.module.inner_loop_main_process()
            assert False
        except NotImplementedError:
            assert True

    def test_inner_loop_post_process(self):
        try:
            self.module.inner_loop_post_process()
            assert False
        except NotImplementedError:
            assert True

    def test_serialize(self):
        try:
            self.module._serialize()
            assert False
        except NotImplementedError:
            assert True

    def test_deserialize(self):
        try:
            self.module._deserialize({})
            assert False
        except NotImplementedError:
            assert True
