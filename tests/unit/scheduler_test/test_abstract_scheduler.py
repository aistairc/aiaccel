import asyncio
import os
import time
from unittest.mock import patch

import numpy as np

from aiaccel.scheduler import AbstractScheduler
from aiaccel.optimizer import create_optimizer

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
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        assert type(AbstractScheduler(config, optimizer)) is AbstractScheduler

    def test_pre_process(
        self,
        setup_hp_running,
        setup_result,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        scheduler = AbstractScheduler(config, optimizer)
        setup_hp_running(2)
        setup_result(1)

        scheduler.pre_process()

        scheduler = AbstractScheduler(config, optimizer)
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

        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        scheduler = AbstractScheduler(config, optimizer)
        assert scheduler.post_process() is None

        with patch.object(scheduler, 'jobs', jobs):
            assert scheduler.post_process() is None

        assert scheduler.post_process() is None

    def test_inner_loop_main_process(
        self,
        clean_work_dir,
        config_json,
        setup_hp_ready,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        scheduler = AbstractScheduler(config, optimizer)
        scheduler.pre_process()
        setup_hp_ready(1)
        assert scheduler.inner_loop_main_process()

    def test_serialize(
        self,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        scheduler = AbstractScheduler(config, optimizer)
        scheduler._rng = np.random.RandomState(0)
        scheduler.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        assert scheduler.serialize(trial_id=0) is None

    def test_deserialize(
        self,
        clean_work_dir,
        config_json,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        scheduler = AbstractScheduler(config, optimizer)
        scheduler.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        scheduler._rng = np.random.RandomState(0)
        scheduler.serialize(trial_id=0)
        assert scheduler.deserialize(trial_id=0) is None

    def test_is_error_free(self, config_json, database_remove):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        scheduler = AbstractScheduler(config, optimizer)
        assert scheduler.is_error_free() is True

        jobstates = [
            {'trial_id': 0, 'jobstate': 'failure'}
        ]

        with patch.object(scheduler, 'job_status', {1: 'failure'}):
            with patch.object(scheduler.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert scheduler.is_error_free() is True

        with patch.object(scheduler, 'job_status', {0: 'failure'}):
            with patch.object(scheduler.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert scheduler.is_error_free() is False

    def test_resume(self, config_json):
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        scheduler = AbstractScheduler(config, optimizer)
        scheduler.pre_process()
        scheduler.serialize(0)
        scheduler.serialize(1)

        scheduler.config.resume = 0
        assert scheduler.resume() is None

        scheduler.config.resume = None
        assert scheduler.resume() is None
