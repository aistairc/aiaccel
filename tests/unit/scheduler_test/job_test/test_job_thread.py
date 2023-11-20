import asyncio
import datetime
import time
from subprocess import Popen

import pytest

from aiaccel.common import (
    dict_hp_finished,
    dict_hp_ready,
    dict_hp_running,
    dict_lock,
    dict_result,
    dict_runner,
    goal_maximize,
    goal_minimize,
)
from aiaccel.config import ResourceType
from aiaccel.scheduler import AbciModel, CustomMachine, Job, LocalModel, LocalScheduler, create_scheduler
from aiaccel.util.process import OutputHandler
from tests.base_test import BaseTest


async def async_start_job(job):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, job.start())


async def async_stop_job_after_sleep(job, sleep_time):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    job.get_machine().set_state('Sucess')
    job.join()


class TestModel(BaseTest):

    @ pytest.fixture(autouse=True)
    def setup_job(
        self,
        clean_work_dir,
        config_json,
        load_test_config,
        setup_hp_ready,
        work_dir
    ):

        self.workspace.clean()
        self.workspace.create()

        config = self.load_config_for_test(self.configs['config.json'])
        scheduler = create_scheduler(config.resource.type.value)(config)

        setup_hp_ready(1)
        trial_id = 0
        self.job = Job(
            config,
            scheduler,
            scheduler.create_model(),
            trial_id
        )
        self.model = scheduler.create_model()
        yield
        self.job = None
        self.model = None

    @pytest.fixture
    def setup_abci_job(
        self,
        config_json,
        work_dir,
        database_remove
    ):
        config = self.load_config_for_test(self.configs['config.json'])
        config.resource.type = ResourceType('abci')

        scheduler = create_scheduler(config.resource.type.value)(config)

        trial_id = 1
        self.abci_job = Job(
            config,
            scheduler,
            scheduler.create_model(),
            trial_id
        )
        yield
        self.abci_job = None

    def test_after_running(self, database_remove):
        assert self.model.after_running(self.job) is None

    def test_after_finished(self, database_remove):
        assert self.model.after_finished(self.job) is None

    def test_before_finished(
        self,
        setup_hp_running,
        setup_result,
        work_dir,
        database_remove
    ):

        for i in range(10):
            self.job.write_start_time_to_storage()
            self.job.storage.trial.set_any_trial_state(trial_id=i, state='finished')
            self.job.storage.hp.set_any_trial_params(
                trial_id=i,
                params=[
                    {'parameter_name': f'x{j+1}', 'value': 0.0, 'type': 'uniform_float'}
                    for j in range(10)
                ]
            )
        assert self.model.before_finished(self.job) is None


class TestJob(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_job(
        self,
        clean_work_dir,
        config_json,
        load_test_config,
        setup_hp_ready,
        work_dir
    ):
        self.workspace.clean()
        self.workspace.create()

        config = self.load_config_for_test(self.configs['config.json'])
        scheduler = create_scheduler(config.resource.type.value)(config)

        setup_hp_ready(1)
        trial_id = 1
        self.job = Job(
            config,
            scheduler,
            scheduler.create_model(),
            trial_id
        )
        yield
        self.job = None

    def test_init(
        self,
        clean_work_dir,
        config_json,
        load_test_config,
        setup_hp_ready,
        work_dir,
        database_remove
    ):
        config = self.load_config_for_test(self.configs['config.json'])
        scheduler = LocalScheduler(config)
        # config = load_test_config()
        setup_hp_ready(1)
        trial_id = 1
        job = Job(
            config,
            scheduler,
            scheduler.create_model(),
            trial_id
        )
        assert type(job) is Job


    def test_get_state_name(self, database_remove):
        assert self.job.get_state_name() == 'ready'
