import asyncio
import os
import time
from unittest.mock import patch

import aiaccel
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler

from tests.base_test import BaseTest


async def async_function(func):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, func)


async def make_directory(sleep_time, d):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    os.mkdir(d)


async def stop_jobs(sleep_time, scheduler):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    for job in scheduler.jobs:
        machine = job['thread'].get_machine()
        machine.set_state('Success')


'''
def test_create_stat():
    assert create_stat(1, 'name') == {
        'job-ID': 1, 'prior': None, 'name': 'name', 'user': None, 'state': 'r',
        'submit/start at': None, 'queue': None, 'jclass': '', 'slots': None,
        'ja-task-ID': None
    }
'''


class TestAbstractScheduler(BaseTest):
    # TODO: Need to fix?
    #  logging modules are generated each method.

    def test_init(self, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        assert type(AbstractScheduler(options)) is AbstractScheduler

    def test_change_state_finished_trials(
        self,
        clean_work_dir,
        config_json,
        setup_hp_running,
        setup_result,
        work_dir,
        database_remove
    ):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        database_remove()
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        setup_hp_running(1)
        setup_result(1)
        assert scheduler.change_state_finished_trials() is None

    def test_get_stats(
        self,
        config_json,
        work_dir,
        database_remove
    ):
        database_remove()
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        assert scheduler.get_stats() is None

    def test_start_job_thread(
        self,
        clean_work_dir,
        config_json,
        setup_hp_ready,
        work_dir,
        database_remove
    ):
        database_remove()
        options = {
            'config': str(config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        setup_hp_ready(1)
        trial_id = 1
        scheduler.start_job_thread(trial_id)
        assert scheduler.start_job_thread(trial_id) is None

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Success')
            job['thread'].join()

    def test_update_resource(
        self,
        config_json,
        work_dir,
        database_remove
    ):
        database_remove()
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        assert scheduler.update_resource() is None

    def test_pre_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_running,
        setup_result,
        work_dir,
        database_remove
    ):
        database_remove()
        options = {
            'config': str(config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.print_dict_state()
        setup_hp_running(2)
        setup_result(1)

        scheduler.pre_process()

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Success')
            job['thread'].join()

    def test_post_process(self, config_json, database_remove):
        database_remove()
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        assert scheduler.post_process() is None

    def test_inner_loop_main_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_ready,
        database_remove
    ):
        database_remove()
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.pre_process()
        setup_hp_ready(1)

        assert scheduler.inner_loop_main_process()

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Scheduling')

        assert scheduler.inner_loop_main_process()

        for job in scheduler.jobs:
            machine = job['thread'].get_machine()
            machine.set_state('Success')
            job['thread'].join()

    def test_serialize(
        self,
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
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        assert scheduler._serialize(trial_id=0) is None

    def test_deserialize(
        self,
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
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        scheduler.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        scheduler._serialize(trial_id=0)
        assert scheduler._deserialize(trial_id=0) is None

    def test_parse_trial_id(self, config_json, database_remove):
        database_remove()
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = AbstractScheduler(options)
        s = "python wrapper.py --trial_id 001"
        s = {"name": "2 python user.py --trial_id 5 --config config.yaml --x1=1.0 --x2=1.0"}
        trial_id = scheduler.parse_trial_id(s['name'])
        # assert name in '001'
        assert trial_id is None
