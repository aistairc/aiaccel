import asyncio
import logging
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest

import aiaccel
from aiaccel.master.local_master import LocalMaster
from aiaccel.module import AbstractModule
from aiaccel.optimizer.random_optimizer import RandomOptimizer
from aiaccel.scheduler.local_scheduler import LocalScheduler

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


# def test_make_work_directory_exit(config_json, work_dir):
#     options = {
#         'config': config_json,
#         'process_name': 'test'
#     }
#     module = AbstractModule(options)
#     module.logger = logging.getLogger(__name__)
#     shutil.rmtree(work_dir)
#     file_create(work_dir.parent.joinpath('work'), "")

#     try:
#         module.make_work_directory()
#         assert False
#     except NotADirectoryError:
#         assert True


class TestAbstractModule(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_module(self):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            # 'fs': False,
            'process_name': 'test',
            'logger_name': __name__
        }

        self.module = AbstractModule(options)
        yield
        self.module = None

    def test_get_each_state_count(self):
        assert self.module.get_each_state_count() is None
        assert self.module.hp_ready == 0
        assert self.module.hp_running == 0
        assert self.module.hp_finished == 0

    def test_get_module_type(self):
        module_type = self.module.get_module_type()
        assert module_type is None

        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            # 'fs': False,
            # 'process_name': 'master'
        }
        commandline_args = [
            "start.py",
            "--config",
            str(self.config_json)
        ]

        with patch.object(sys, 'argv', commandline_args):
            master = LocalMaster(options)
            module_type = master.get_module_type()
            assert module_type == aiaccel.module_type_master

            options = {
                'config': str(self.config_json),
                'resume': None,
                'clean': False,
                # 'fs': False,
                # 'process_name': 'optimizer'
            }
            optimizer = RandomOptimizer(options)
            module_type = optimizer.get_module_type()
            assert module_type == aiaccel.module_type_optimizer

            options = {
                'config': str(self.config_json),
                'resume': None,
                'clean': False,
                # 'fs': False,
                # 'process_name': 'scheduler'
            }
            scheduler = LocalScheduler(options)
            module_type = scheduler.get_module_type()
            assert module_type == aiaccel.module_type_scheduler

    def test_check_finished(self, setup_hp_finished):
        assert not self.module.check_finished()

        setup_hp_finished(
            # int(self.module.config.get('hyperparameter', 'trial_number'))
            # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
            int(self.module.config.trial_number.get())
        )

        assert self.module.check_finished()

    def test_print_dict_state(self):
        assert self.module.print_dict_state() is None

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
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'process_name': 'test',
            'logger_name': 'root.optimizer'
        }

        self.module = AbstractModule(options)
        self.module._rng = np.random.RandomState(0)
        assert self.module.resume() is None

        self.module.options['resume'] = 1
        self.module._serialize(1)
        assert self.module.resume() is None

    def test_create_log_file_handler(self, work_dir) -> None:
        log_file = work_dir.joinpath(
            self.module.dict_log,
            self.module.config.optimizer_logfile.get()
        )
        file_level = self.module.config.optimizer_file_log_level.get()
        file_handler = self.module._create_log_file_handler(log_file, file_level)
        assert file_handler.baseFilename == str(log_file)
        assert file_handler.level == logging.__getattribute__(file_level)
        assert file_handler.formatter._fmt == (
            '%(asctime)s %(levelname)-8s %(filename)-12s line '
            '%(lineno)-4s %(message)s'
        )

    def test_create_stream_file_handler(self):
        stream_level = self.module.config.optimizer_stream_log_level.get()
        stream_handler = self.module._create_log_stream_handler(stream_level)
        assert stream_handler.level == logging.__getattribute__(stream_level)
        assert stream_handler.formatter._fmt == 'Test      %(levelname)-8s %(message)s'

    def test_set_log_handlers(self, work_dir):
        log_file = work_dir.joinpath(
            self.module.dict_log,
            self.module.config.optimizer_logfile.get()
        )
        file_level = self.module.config.optimizer_file_log_level.get()
        stream_level = self.module.config.optimizer_stream_log_level.get()
        assert self.module._set_log_handlers(log_file, file_level, stream_level) is None
        assert self.module._logger.name == __name__
        assert self.module._logger.handlers[0].baseFilename == str(log_file)
        if isinstance(file_level, str):
            assert self.module._logger.handlers[0].level == logging.__getattribute__(file_level)
        elif isinstance(file_level, int):
            assert self.module._logger.handlers[0].level == file_level
        else:
            assert False
        assert self.module._logger.handlers[0].formatter._fmt == (
            '%(asctime)s %(levelname)-8s %(filename)-12s line '
            '%(lineno)-4s %(message)s'
        )
        if isinstance(stream_level, str):
            assert self.module._logger.handlers[0].level == logging.__getattribute__(stream_level)
        elif isinstance(stream_level, int):
            assert self.module._logger.handlers[0].level == file_level
        else:
            assert False
        assert self.module._logger.handlers[1].formatter._fmt == 'Test      %(levelname)-8s %(message)s'
