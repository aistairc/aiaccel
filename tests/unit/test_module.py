from aiaccel.master.local import LocalMaster
from aiaccel.module import AbstractModule
from aiaccel.optimizer.random.search import RandomSearchOptimizer
from aiaccel.scheduler.local import LocalScheduler
from aiaccel.util.filesystem import file_create
from aiaccel.util.logger import str_to_logging_level
from contextlib import ExitStack
from pathlib import Path
from tests.base_test import BaseTest
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
import aiaccel
import asyncio
import logging
import pytest
import shutil
import time
import sys
from aiaccel.storage.storage import Storage


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
    def setup_module(self, clean_work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'test'
        }

        self.module = AbstractModule(options)
        self.module.storage.alive.init_alive()
        self.module.logger = logging.getLogger(__name__)
        yield
        self.module = None

    # def test_make_work_directory(self, work_dir):
    #     assert self.module.make_work_directory() is None

    #     file_create(work_dir.joinpath(aiaccel.dict_state, 'test.txt'), "")
    #     try:
    #         self.module.make_work_directory()
    #         assert True
    #     except NotImplementedError:
    #         assert False

    #     work_dir.joinpath(aiaccel.dict_state, 'test.txt').unlink()
    #     shutil.rmtree(work_dir.joinpath(aiaccel.dict_lock))
    #     file_create(work_dir.joinpath(aiaccel.dict_lock), "")
    #     assert self.module.make_work_directory() is None

    # def test_check_work_directory(self, work_dir):
    #     shutil.rmtree(work_dir.joinpath(aiaccel.dict_runner))
    #     assert not self.module.check_work_directory()
    #     work_dir.joinpath(aiaccel.dict_runner).mkdir()
    #     assert self.module.check_work_directory()

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
            'fs': False,
            'process_name': 'master'
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
                'fs': False,
                'process_name': 'optimizer'
            }
            optimizer = RandomSearchOptimizer(options)
            module_type = optimizer.get_module_type()
            assert module_type == aiaccel.module_type_optimizer

            options = {
                'config': str(self.config_json),
                'resume': None,
                'clean': False,
                'fs': False,
                'process_name': 'scheduler'
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

    def test_exit_alive(self, work_dir):

        assert self.module.exit_alive('master') is None

        # with pytest.raises(SystemExit) as e:
        #     self.module.exit_alive('master')

        # assert e.type == SystemExit
        # assert e.value.code is None

    def test_print_dict_state(self):
        assert self.module.print_dict_state() is None

    def test_set_logger(self, work_dir):
        assert self.module.set_logger(
            'root.optimizer',
            work_dir.joinpath(
                self.module.dict_log,
                # self.config.get('logger', 'optimizer_logfile')
                # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(2021-08-12:荒本)
                self.module.config.optimizer_logfile.get()
            ),
            str_to_logging_level(
                # self.module.config.get('logger', 'optimizer_file_log_level')
                # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(2021-08-12:荒本)
                self.module.config.optimizer_file_log_level.get()
            ),
            str_to_logging_level(
                # self.module.config.get('logger', 'optimizer_stream_log_level')
                # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
                self.module.config.optimizer_stream_log_level.get()
            ),
            'Optimizer'
        ) is None

    def test_pre_process(self, work_dir):
        self.module.get_module_type = MagicMock(return_value=(aiaccel.module_type_optimizer))
        self.module.storage.alive.init_alive()
        assert self.module.pre_process() is None

        with pytest.raises(SystemExit) as e:
            self.module.pre_process()

        assert e.type == SystemExit
        assert e.value.code is None

    def test_post_process(self):
        try:
            self.module.post_process()
            assert False
        except NotImplementedError:
            assert True

    def test_start(self):
        with patch.object(self.module, "loop", return_value=True),\
                patch.object(self.module, 'pre_process', return_value=True),\
                patch.object(self.module, 'post_process', return_value=True):
            assert self.module.start() is None

    def test_loop_pre_process(self, work_dir):
        # work_dir.joinpath(aiaccel.dict_runner).rmdir()
        # loop = asyncio.get_event_loop()
        # gather = asyncio.gather(
        #     async_function(self.module.loop_pre_process),
        #     delay_make_directory(1, work_dir.joinpath(aiaccel.dict_runner))
        # )
        # loop.run_until_complete(gather)
        # assert self.module.loop_pre_process() is None
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

    def test_loop(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                self.module, 'loop_pre_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_pre_process', return_value=False
            ))
            stack.enter_context(patch.object(
                self.module, 'loop_post_process', return_value=True
            ))
            assert self.module.loop() is None

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                self.module, 'loop_pre_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_pre_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'loop_post_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_main_process', return_value=False
            ))
            assert self.module.loop() is None

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                self.module, 'loop_pre_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_pre_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_main_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_post_process', return_value=False
            ))
            stack.enter_context(patch.object(
                self.module, 'loop_post_process', return_value=True
            ))
            assert self.module.loop() is None

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                self.module, 'loop_pre_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_pre_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_main_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'inner_loop_post_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'loop_post_process', return_value=True
            ))
            stack.enter_context(patch.object(
                self.module, 'check_error', return_value=False
            ))
            self.module._serialize = Mock()
            self.module._serialize.side_effect = dummy_break

            self.module.loop()

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

    def test_is_process_alive(self):
        assert not self.module.is_process_alive()
