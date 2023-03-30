from __future__ import annotations

import os
import time
from collections.abc import Callable
from collections.abc import Generator
from unittest.mock import patch

import pytest
import asyncio
import numpy as np

from aiaccel.command_line_options import CommandLineOptions
from aiaccel.scheduler import AbstractScheduler
from aiaccel.scheduler import LocalModel

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

    @pytest.fixture(autouse=True)
    def setup_scheduler(
        self,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
        database_remove: Callable[[None], None]
    ) -> Generator[None, None, None]:
        self.options = CommandLineOptions(
            config=str(self.config_json),
            resume=None,
            clean=False,
            process_name="scheduler"
        )
        database_remove()
        if "noautousefixture" in request.keywords:
            yield
        else:
            self.scheduler = AbstractScheduler(self.options)
            yield
        self.options = None
        self.scheduler = None

    @pytest.mark.noautousefixture
    def test_init(self) -> None:
        assert type(AbstractScheduler(self.options)) is AbstractScheduler

    def test_change_state_finished_trials(
        self,
        setup_hp_running,
        setup_result,
    ) -> None:
        setup_hp_running(1)
        setup_result(1)
        assert self.scheduler.change_state_finished_trials() is None

    def test_get_stats(self) -> None:
        assert self.scheduler.get_stats() is None

    def test_start_job(self, setup_hp_ready) -> None:
        setup_hp_ready(1)
        trial_id = 1
        self.scheduler.start_job(trial_id)
        assert self.scheduler.start_job(trial_id) is None

        for job in self.scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Success')
            job.main()

    def test_update_resource(self) -> None:
        assert self.scheduler.update_resource() is None

    def test_pre_process(
        self,
        setup_hp_running,
        setup_result,
    ) -> None:
        setup_hp_running(2)
        setup_result(1)
        self.scheduler.pre_process()

        for job in self.scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Success')
            job.main()

        scheduler = AbstractScheduler(self.options)
        with patch.object(scheduler.storage.trial, 'get_running', return_value=[]):
            assert scheduler.pre_process() is None

        with patch.object(scheduler.storage.trial, 'get_running', return_value=[0, 1, 2]):
            assert scheduler.pre_process() is None

    def test_post_process(self) -> None:
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

        scheduler = AbstractScheduler(self.options)
        assert scheduler.post_process() is None

        with patch.object(scheduler, 'check_finished', return_value=False):
            with patch.object(scheduler, 'jobs', jobs):
                assert scheduler.post_process() is None

        with patch.object(scheduler, 'check_finished', return_value=True):
            assert scheduler.post_process() is None

    def test_inner_loop_main_process(
        self,
        setup_hp_ready
    ):
        self.scheduler.pre_process()
        setup_hp_ready(1)

        assert self.scheduler.inner_loop_main_process()

        for job in self.scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Scheduling')

        assert self.scheduler.inner_loop_main_process()

        for job in self.scheduler.jobs:
            machine = job.get_machine()
            machine.set_state('Success')
            job.main()

        with patch.object(self.scheduler, 'check_finished', return_value=True):
            assert self.scheduler.inner_loop_main_process() is False

        with patch.object(self.scheduler, 'all_done', return_value=True):
            assert self.scheduler.inner_loop_main_process() is False

    def test_serialize(
        self,
        monkeypatch: pytest.MonkeyPatch
    ):
        with monkeypatch.context() as m:
            m.setattr(self.scheduler, "_rng", np.random.RandomState(0))
            m.setattr(
                self.scheduler.storage.trial,
                "get_any_trial_state",
                lambda trial_id: "finished" if trial_id == 0 else ""
            )
            assert self.scheduler._serialize(trial_id=0) is None

    def test_deserialize(
        self,
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.scheduler, "_rng", np.random.RandomState(0))
            m.setattr(
                self.scheduler.storage.trial,
                "get_any_trial_state",
                lambda trial_id: "finished" if trial_id == 0 else ""
            )
            self.scheduler._serialize(trial_id=0)
            assert self.scheduler._deserialize(trial_id=0) is None

    def test_parse_trial_id(self) -> None:
        s = "python wrapper.py --trial_id 001"
        s = {"name": "2 python user.py --trial_id 5 --config config.yaml --x1=1.0 --x2=1.0"}
        trial_id = self.scheduler.parse_trial_id(s['name'])
        # assert name in '001'
        assert trial_id is None

    def test_check_error(self) -> None:
        assert self.scheduler.check_error() is True

        jobstates = [
            {'trial_id': 0, 'jobstate': 'failure'}
        ]

        with patch.object(self.scheduler, 'job_status', {1: 'failure'}):
            with patch.object(self.scheduler.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert self.scheduler.check_error() is True

        with patch.object(self.scheduler, 'job_status', {0: 'failure'}):
            with patch.object(self.scheduler.storage.jobstate, 'get_all_trial_jobstate', return_value=jobstates):
                assert self.scheduler.check_error() is False

    def test_resume(self) -> None:
        self.scheduler.pre_process()
        self.scheduler._serialize(0)
        self.scheduler._serialize(1)

        self.scheduler.options.resume = 1
        assert self.scheduler.resume() is None

        self.scheduler.options.resume = None
        assert self.scheduler.resume() is None

    def test_create_model(self) -> None:
        assert type(self.scheduler.create_model()) is LocalModel
