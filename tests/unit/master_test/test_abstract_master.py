from aiaccel import workspace
from aiaccel.config import ConfileWrapper, Config
from aiaccel.master.abstract import AbstractMaster
from aiaccel.util.filesystem import get_dict_files
from aiaccel.util.time_tools import get_time_now_object
from tests.base_test import BaseTest
from unittest.mock import patch
import aiaccel
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
import sys
from aiaccel.master.create import create_master
from aiaccel.argument import Arguments
from aiaccel.workspace import Workspace
from functools import wraps


async def loop_pre_process(master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, master.pre_process)


async def delay_make_directory(sleep_time, d):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    os.mkdir(d)
    time.sleep(10)
    return


def callback_return():
    return



class TestAbstractMaster(BaseTest):

    def test_pre_process(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        work_dir,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]

        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            # master = create_master(options['config'])(options)
            master = AbstractMaster(options)
            # master = start.Master()
        # master = AbstractMaster(options)
        work_dir.joinpath(aiaccel.dict_runner).rmdir()
        loop = asyncio.get_event_loop()
        gather = asyncio.gather(
            loop_pre_process(master),
            delay_make_directory(1, work_dir.joinpath(aiaccel.dict_runner))
        )
        loop.run_until_complete(gather)

        master.storage.alive.init_alive()

        if master.scheduler_proc is not None:
            master.scheduler_proc.wait()

        if master.optimizer_proc is not None:
            master.optimizer_proc.wait()

        master.worker_o.kill()
        master.worker_s.kill()
        master.storage.alive.init_alive()

    def test_pre_process_2(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        fake_process,
        work_dir,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = AbstractMaster(options)
        master.start_optimizer()
        master.start_scheduler()

        # # master = AbstractMaster(config_json)
        # opt_cmd = master.config.optimizer_command.get().split(" ")
        # opt_cmd.append('--config')
        # opt_cmd.append(str(config_json))

        # # opt_cmd = [
        # #     # 'python',
        # #     # '-m',
        # #     # master.config.get('optimizer', 'optimizer_command'),
        # #     master.config.optimizer_command.get(),
        # #     '--config',
        # #     config_json
        # # ]
        # fake_process.register_subprocess(
        #     opt_cmd, callback=callback_module
        # )
        # sch_cmd = master.config.scheduler_command.get().split(" ")
        # sch_cmd.append('--config')
        # sch_cmd.append(str(config_json))
        # # sch_cmd = [
        # #     # 'python',
        # #     # '-m',
        # #     # master.config.get('scheduler', 'scheduler_command'),
        # #     master.config.scheduler_command.get(),
        # #     '--config',
        # #     config_json
        # # ]
        # fake_process.register_subprocess(
        #     sch_cmd, callback=callback_module
        # )
        try:
            master.pre_process()
            assert False
        except AssertionError:
            assert True

        # master.th_optimizer.abort()
        # master.th_scheduler.abort()
        master.worker_o.kill()
        master.worker_s.kill()
        master.storage.alive.init_alive()

    def test_pre_process_3(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        setup_hp_finished,
        work_dir,
        database_remove
    ):
        database_remove()
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        master = AbstractMaster(options)
        setup_hp_finished(10)
        assert master.pre_process() is None

        master.storage.alive.init_alive()

        if master.scheduler_proc is not None:
            master.scheduler_proc.wait()

        if master.optimizer_proc is not None:
            master.optimizer_proc.wait()

        master.worker_o.kill()
        master.worker_s.kill()
        master.storage.alive.init_alive()

    def test_post_process(
        self,
        cd_work,
        clean_work_dir,
        setup_hp_finished,
        work_dir,
        database_remove
    ):
        database_remove()
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        master = AbstractMaster(options)
        master.storage.alive.init_alive()
        
        for i in range(10):
            master.storage.trial.set_any_trial_state(trial_id=i, state='finished')
            master.storage.result.set_any_trial_objective(trial_id=i, objective=(i * 10.0))
            for j in range(2):
                master.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f"x{j}",
                    param_value=0.0,
                    param_type='flaot'
                )
        assert master.post_process() is None

        # with open(self.config_json, 'r') as f:
        #     json_obj = json.load(f)
        # json_obj['hyperparameter']['goal'] = aiaccel.goal_maximize
        # config = ConfileWrapper(json_obj, 'json_object')
        # master.config = config
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        master.config = Config(self.config_json)
        master.config.goal.set(aiaccel.goal_maximize)
        assert master.post_process() is None

        # json_obj['hyperparameter']['goal'] = 'invalid_goal'
        # config = ConfileWrapper(json_obj, 'json_object')
        # master.config = config
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        master.config = Config(self.config_json)
        # master.config.goal.set('invalid_goal')
        master.goal = 'invalid_goal'

        for i in range(10):
            master.storage.trial.set_any_trial_state(trial_id=i, state='finished')
            
        try:
            master.post_process()
            assert False
        except ValueError:
            assert True

    def test_print_dict_state(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        setup_hp_finished,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        with patch.object(sys, 'argv', commandline_args):
            # from aiaccel import start
            # master = start.Master()
            options = Arguments()
            master = AbstractMaster(options)

        # master = AbstractMaster(config_json)
        assert master.print_dict_state() is None

        master.loop_start_time = get_time_now_object()
        assert master.print_dict_state() is None

        setup_hp_finished(1)
        master.get_each_state_count()
        assert master.print_dict_state() is None

    def test_start_optimizer(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        fake_process,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        with patch.object(sys, 'argv', commandline_args):
            # from aiaccel import start
            # master = start.Master()
            options = Arguments()
            master = AbstractMaster(options)

        assert master.start_optimizer() is None
        master.worker_o.kill()

        # master = AbstractMaster(config_json)
        # opt_cmd = master.config.optimizer_command.get().split(" ")
        # opt_cmd.append('--config')
        # opt_cmd.append(str(config_json))

        # fake_process.register_subprocess(
        #     master.start_optimizer(), callback=callback_return
        # )
        # assert master.start_optimizer() is None
        # master.th_optimizer.abort()

    def test_start_scheduler(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        fake_process,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = AbstractMaster(options)

        assert master.start_scheduler() is None
        master.worker_s.kill()

        # master = AbstractMaster(config_json)
        # sch_cmd = master.config.scheduler_command.get().split(" ")
        # sch_cmd.append('--config')
        # sch_cmd.append(str(config_json))
        # # sch_cmd = [
        # #     'python',
        # #     '-m',
        # #     master.config.get('scheduler', 'scheduler_command'),
        # #     '--config',
        # #     config_json
        # # ]
        # fake_process.register_subprocess(
        #     sch_cmd, callback=callback_return
        # )
        # assert master.start_scheduler() is None
        # master.th_scheduler.abort()

    def test_loop_pre_process(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()

        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        with patch.object(sys, 'argv', commandline_args):
            master = AbstractMaster(options)
        assert master.loop_pre_process() is None

    def test_loop_post_process(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        with patch.object(sys, 'argv', commandline_args):
            master = AbstractMaster(options)
        p = subprocess.Popen(['ls'])
        with patch.object(master, 'optimizer_proc', return_value=p):
            with patch.object(master, 'scheduler_proc', return_value=p):
                assert master.loop_post_process() is None

    def test_inner_loop_pre_process(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()

        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        with patch.object(sys, 'argv', commandline_args):
            master = AbstractMaster(options)
        master.storage.alive.init_alive()
        # with patch.object(master, 'ws', return_value='/'):
        #     with patch('aiaccel.dict_alive', return_value=''):
        #         with patch('aiaccel.alive_master', return_value='tmp'):
        #             with patch.object(master, 'get_each_state_count', return_value=None):
        #                 master.pre_process()
        #                 assert master.inner_loop_pre_process()
        with patch.object(master, 'ws', return_value='/tmp'):
            with patch.object(master, 'get_each_state_count', return_value=None):
                master.storage.alive.init_alive()
                master.pre_process()
                assert master.inner_loop_pre_process()

        master.worker_o.kill()
        master.worker_s.kill()
        master.storage.alive.init_alive()

    def test_inner_loop_main_process(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        setup_hp_finished,
        database_remove
    ):
        database_remove()
        # options = {
        #     'config': config_json,
        #     'resume': None,
        #     'clean': False,
        #     'fs': False,
        #     'process_name': 'master'
        # }
        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = AbstractMaster(options)
        
        master.storage.alive.init_alive()
        master.pre_process()
        master.inner_loop_pre_process()
        assert master.inner_loop_main_process()

        master.trial_number = 10
        for i in range(10):
            master.storage.trial.set_any_trial_state(trial_id=i, state='finished')
        # setup_hp_finished(10)
        master.get_each_state_count()
        assert not master.inner_loop_main_process()

        master.worker_o.kill()
        master.worker_s.kill()
        master.storage.alive.init_alive()

    def test_inner_loop_post_process(
        self,
        cd_work,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()

        commandline_args = [
            "start.py",
            "--config",
            format(config_json)
        ]
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        with patch.object(sys, 'argv', commandline_args):
            master = AbstractMaster(options)
        master = AbstractMaster(options)
        master.storage.alive.init_alive()

        master.pre_process()
        master.inner_loop_pre_process()
        master.inner_loop_main_process()
        assert master.inner_loop_post_process()

        master.worker_o.kill()
        master.worker_s.kill()
        master.storage.alive.init_alive()
