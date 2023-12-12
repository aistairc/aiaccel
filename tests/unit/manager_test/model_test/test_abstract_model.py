import time
import datetime
import pytest

from aiaccel.config import ResourceType
from aiaccel.manager import CustomMachine, Job, LocalModel, LocalManager, create_manager, AbstractManager
from aiaccel.manager.job.model import AbstractModel
from aiaccel.util.process import OutputHandler
from tests.base_test import BaseTest
from aiaccel.optimizer import create_optimizer
from aiaccel.common import datetime_format


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
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = create_manager(config.resource.type.value)(config, optimizer)

        setup_hp_ready(1)
        trial_id = 0
        self.job = Job(
            config,
            manager,
            manager.create_model(),
            trial_id
        )
        self.model = AbstractModel()
        yield
        self.job = None
        self.model = None

    def test_runner_create(self, database_remove):
        assert self.model.runner_create(self.job) is None

    def test_after_running(self, database_remove):
        assert self.model.after_running(self.job) is None

    def test_before_timeout(self, database_remove):
        trial_id = 0
        start_time = datetime.datetime(1900, 1, 1, 0, 0, 0, 0).strftime(datetime_format)
        end_time = datetime.datetime.now().strftime(datetime_format)
        self.job.manager.storage.timestamp.set_any_trial_start_time(trial_id, start_time)
        self.job.manager.storage.timestamp.set_any_trial_end_time(trial_id, end_time)
        assert self.model.before_timeout(self.job) is None

    def test_after_timeout(self, database_remove):
        assert self.model.after_timeout(self.job) is None

    def test_stop_job(self, database_remove):
        assert self.model.stop_job(self.job) is None

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
