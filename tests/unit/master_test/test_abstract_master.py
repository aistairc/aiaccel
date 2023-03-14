
import asyncio
import os
import sys
import time
from unittest.mock import patch

from aiaccel.common import goal_maximize
from aiaccel.config import Config
from aiaccel.master import AbstractMaster
from aiaccel.util import get_time_now_object
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
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(self.config_json)
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
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(self.config_json)
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
        setup_hp_finished,
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
        master.config.goal.set(goal_maximize)
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
        setup_hp_finished,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(self.config_json)
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

    def test_inner_loop_main_process(
        self,
        database_remove
    ):
        database_remove()
        commandline_args = [
            "start.py",
            "--config",
            format(self.config_json)
        ]
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        with patch.object(sys, 'argv', commandline_args):
            options = parse_arguments()
            master = AbstractMaster(options)

        master.pre_process()
        assert master.inner_loop_main_process()

        master.trial_number = 10
        for i in range(10):
            master.storage.trial.set_any_trial_state(trial_id=i, state='finished')
        # setup_hp_finished(10)
        master.get_each_state_count()
        assert not master.inner_loop_main_process()
