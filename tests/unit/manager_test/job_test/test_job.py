import asyncio
import datetime
import time
from subprocess import Popen

import pytest

from aiaccel.common import (
    dict_lock,
    dict_runner,
    goal_maximize,
    goal_minimize,
)
from aiaccel.config import ResourceType
from aiaccel.manager import CustomMachine, Job, LocalModel, LocalManager, create_manager
from aiaccel.util.process import OutputHandler
from tests.base_test import BaseTest
from aiaccel.optimizer import create_optimizer


# async def async_start_job(job):
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, job.start())


# async def async_stop_job_after_sleep(job, sleep_time):
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, time.sleep, sleep_time)
#     job.get_machine().set_state('Sucess')
#     job.join()


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
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = create_manager(config.resource.type.value)(config, optimizer)

        setup_hp_ready(1)
        trial_id = 1
        self.job = Job(
            config,
            manager,
            manager.create_model(),
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
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        # config = load_test_config()
        setup_hp_ready(1)
        trial_id = 1
        job = Job(
            config,
            manager,
            manager.create_model(),
            trial_id
        )
        assert type(job) is Job


    def test_get_state_name(self, database_remove):
        assert self.job.get_state_name() == 'ready'
