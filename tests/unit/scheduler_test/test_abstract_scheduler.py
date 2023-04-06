import asyncio
import os
import time
from unittest.mock import patch

import numpy as np

from aiaccel.scheduler import AbstractScheduler

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
        machine = job['obj'].get_machine()
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
        assert type(AbstractScheduler(self.load_config_for_test(self.configs['config.json']))) is AbstractScheduler

    def test_change_state_finished_trials(
        self,
        setup_hp_running,
        setup_result,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.print_dict_state()
        setup_hp_running(1)
        setup_result(1)
        assert scheduler.change_state_finished_trials() is None

    def test_get_stats(
        self,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.print_dict_state()
        assert scheduler.get_stats() is None

    def test_start_job(
        self,
        setup_hp_ready,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.print_dict_state()
        setup_hp_ready(1)
        trial_id = 1
        scheduler.start_job(trial_id)
        assert scheduler.start_job(trial_id) is None

        for job in scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Success')
            job.main()

    def test_update_resource(
        self,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.print_dict_state()
        assert scheduler.update_resource() is None

    def test_pre_process(
        self,
        setup_hp_running,
        setup_result,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.print_dict_state()
        setup_hp_running(2)
        setup_result(1)

        scheduler.pre_process()

        for job in scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Success')
            job.main()

        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        with patch.object(scheduler.storage.trial, 'get_running', return_value=[]):
            assert scheduler.pre_process() is None

        with patch.object(scheduler.storage.trial, 'get_running', return_value=[0, 1, 2]):
            assert scheduler.pre_process() is None

    def test_post_process(self, database_remove):
        database_remove()
        class dummy_job:
            def __init__(self):
                pass

            def stop(self):
                pass

            def join(self):
                pass

        jobs = []
        for i in range(10):
            jobs.append({'thread': dummy_job()})

        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        assert scheduler.post_process() is None

        with patch.object(scheduler, 'check_finished', return_value=False):
            with patch.object(scheduler, 'jobs', jobs):
                assert scheduler.post_process() is None

        with patch.object(scheduler, 'check_finished', return_value=True):
            assert scheduler.post_process() is None

    def test_inner_loop_main_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_ready,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.pre_process()
        setup_hp_ready(1)

        assert scheduler.inner_loop_main_process()

        for job in scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Scheduling')

        assert scheduler.inner_loop_main_process()

        for job in scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Success')
            job.main()

        with patch.object(scheduler, 'check_finished', return_value=True):
            assert scheduler.inner_loop_main_process() is False

        with patch.object(scheduler, 'all_done', return_value=True):
            assert scheduler.inner_loop_main_process() is False

    def test_serialize(
        self,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler._rng = np.random.RandomState(0)
        scheduler.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        assert scheduler._serialize(trial_id=0) is None

    def test_deserialize(
        self,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        scheduler._rng = np.random.RandomState(0)
        scheduler._serialize(trial_id=0)
        assert scheduler._deserialize(trial_id=0) is None

    def test_parse_trial_id(self, config_json, database_remove):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        s = "python wrapper.py --trial_id 001"
        s = {"name": "2 python user.py --trial_id 5 --config config.yaml --x1=1.0 --x2=1.0"}
        trial_id = scheduler.parse_trial_id(s['name'])
        # assert name in '001'
        assert trial_id is None

    def test_check_error(self, config_json, database_remove):
        database_remove()
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        assert scheduler.check_error() is True

        jobstates = [
            {'trial_id': 0, 'jobstate': 'failure'}
        ]

        with patch.object(scheduler, 'job_status', {1: 'failure'}):
            with patch.object(scheduler.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert scheduler.check_error() is True

        with patch.object(scheduler, 'job_status', {0: 'failure'}):
            with patch.object(scheduler.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert scheduler.check_error() is False

    def test_resume(self, config_json):
        scheduler = AbstractScheduler(self.load_config_for_test(self.configs['config.json']))
        scheduler.pre_process()
        scheduler._serialize(0)
        scheduler._serialize(1)

        scheduler.config.resume = 1
        assert scheduler.resume() is None

        scheduler.config.resume = None
        assert scheduler.resume() is None
