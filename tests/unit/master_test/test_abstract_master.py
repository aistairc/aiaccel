
import asyncio
import json
import os
import subprocess
import sys
import time
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import aiaccel
from aiaccel import workspace
from aiaccel.config import Config, ConfileWrapper
from aiaccel.master.abstract_master import AbstractMaster
from aiaccel.master.create import create_master
from aiaccel.util.filesystem import get_dict_files
from aiaccel.util.time_tools import get_time_now_object
from aiaccel.workspace import Workspace
from tests.arguments import parse_arguments
from tests.base_test import BaseTest


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
            options = parse_arguments()
            master = AbstractMaster(options)
        loop = asyncio.get_event_loop()
        gather = asyncio.gather(
            loop_pre_process(master)
        )
        loop.run_until_complete(gather)

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
            options = parse_arguments()
            master = AbstractMaster(options)

        try:
            master.pre_process()
            assert False
        except AssertionError:
            assert True

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

        master.config = Config(self.config_json)
        master.config.goal.set(aiaccel.goal_maximize)
        assert master.post_process() is None

        master.config = Config(self.config_json)
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
            options = parse_arguments()
            master = AbstractMaster(options)

        # master = AbstractMaster(config_json)
        assert master.print_dict_state() is None

        master.loop_start_time = get_time_now_object()
        assert master.print_dict_state() is None

        setup_hp_finished(1)
        master.get_each_state_count()
        assert master.print_dict_state() is None

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

        with patch.object(master, 'ws', return_value='/tmp'):
            with patch.object(master, 'get_each_state_count', return_value=None):
                master.pre_process()
                assert master.inner_loop_pre_process()

    def test_inner_loop_main_process(
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
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        with patch.object(sys, 'argv', commandline_args):
            options = parse_arguments()
            master = AbstractMaster(options)
        
        master.pre_process()
        master.inner_loop_pre_process()
        assert master.inner_loop_main_process()

        master.trial_number = 10
        for i in range(10):
            master.storage.trial.set_any_trial_state(trial_id=i, state='finished')
        # setup_hp_finished(10)
        master.get_each_state_count()
        assert not master.inner_loop_main_process()

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

        master.pre_process()
        master.inner_loop_pre_process()
        master.inner_loop_main_process()
        assert master.inner_loop_post_process()
