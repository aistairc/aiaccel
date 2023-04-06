import asyncio
import logging
import time

import numpy as np
import pytest
from aiaccel.common import (module_type_master, module_type_optimizer,
                            module_type_scheduler)

from aiaccel.master import LocalMaster
from aiaccel.module import AbstractModule

from aiaccel.optimizer import RandomOptimizer
from aiaccel.scheduler import LocalScheduler
from aiaccel.util import str_to_logging_level
from tests.base_test import BaseTest


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
        self.module = AbstractModule(self.load_config_for_test(self.configs["config.json"]), 'abstract')
        self.module.logger = logging.getLogger(__name__)
        yield
        self.module = None

    def test_update_each_state_count(self):
        assert self.module.update_each_state_count() is None
        assert self.module.hp_ready == 0
        assert self.module.hp_running == 0
        assert self.module.hp_finished == 0

    def test_get_module_type(self):
        module_type = self.module.get_module_type()
        assert module_type is None

        master = LocalMaster(self.load_config_for_test(self.configs["config.json"]))
        module_type = master.get_module_type()
        assert module_type == module_type_master

        optimizer = RandomOptimizer(self.load_config_for_test(self.configs["config.json"]))
        module_type = optimizer.get_module_type()
        assert module_type == module_type_optimizer

        scheduler = LocalScheduler(self.load_config_for_test(self.configs["config.json"]))
        module_type = scheduler.get_module_type()
        assert module_type == module_type_scheduler

    def test_check_finished(self, setup_hp_finished):
        assert not self.module.check_finished()

        setup_hp_finished(
            # int(self.module.config.get('hyperparameter', 'trial_number'))
            # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
            int(self.module.config.optimize.trial_number)
        )

        assert self.module.check_finished()

    def test_print_dict_state(self):
        assert self.module.print_dict_state() is None

    def test_set_logger(self, work_dir):
        assert self.module.set_logger(
            'root.optimizer',
            work_dir.joinpath(
                self.module.workspace.log,
                # self.config.get('logger', 'optimizer_logfile')
                # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(2021-08-12:荒本)
                self.module.config.logger.file.optimizer
            ),
            str_to_logging_level(
                # self.module.config.get('logger', 'optimizer_file_log_level')
                # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(2021-08-12:荒本)
                self.module.config.logger.log_level.optimizer
            ),
            str_to_logging_level(
                # self.module.config.get('logger', 'optimizer_stream_log_level')
                # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
                self.module.config.logger.stream_level.optimizer
            ),
            'Optimizer'
        ) is None

    def test_pre_process(self):
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

    def test_inner_loop_main_process(self):
        try:
            self.module.inner_loop_main_process()
            assert False
        except NotImplementedError:
            assert True

    def test_serialize(self):
        self.module._rng = np.random.RandomState(0)
        assert self.module._serialize(0) is None

    def test_deserialize(self):
        self.module._rng = np.random.RandomState(0)
        self.module._serialize(1)
        assert self.module._deserialize(1) is None

    def test_check_error(self):
        assert self.module.check_error() is True

    def test_resume(self):
        self.module = AbstractModule(self.load_config_for_test(self.configs["config.json"]), 'abstract')
        self.module._rng = np.random.RandomState(0)
        assert self.module.resume() is None

        config = self.load_config_for_test(self.configs["config.json"])
        config.resume = 1
        self.module = AbstractModule(config, 'abstract')
        self.module.set_logger(
            'root.abstract',
            self.module.workspace.log / self.module.config.logger.file.master,
            str_to_logging_level(self.module.config.logger.log_level.master),
            str_to_logging_level(self.module.config.logger.stream_level.master),
            'Abstract   '
        )
        self.module.create_numpy_random_generator()
        self.module._serialize(1)
        assert self.module.resume() is None
