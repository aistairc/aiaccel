import time
import pytest

from aiaccel.config import ResourceType
from aiaccel.manager import (
    CustomMachine,
    Job,
    LocalModel,
    LocalManager,
    create_manager,
    LocalManager
)
from aiaccel.util.process import OutputHandler
from tests.base_test import BaseTest
from aiaccel.optimizer import create_optimizer
from aiaccel.util.job_script_preamble import create_job_script_preamble
from aiaccel.manager.job.model import AbciModel, AbstractModel, LocalModel, MpiModel


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

        self.config = self.load_config_for_test(self.configs['config.json'])
        self.config.resource.type = "abci"
        self.optimizer = create_optimizer(self.config.optimize.search_algorithm)(self.config)
        self.manager = create_manager(self.config.resource.type.value)(self.config, self.optimizer)

        setup_hp_ready(1)
        trial_id = 0
        self.job = Job(
            self.config,
            self.manager,
            self.manager.create_model(),
            trial_id
        )
        self.model = AbciModel()
        yield
        self.job = None
        self.model = None

    def test_create_abci_batch_file(
        self,
        clean_work_dir,
        get_one_parameter,
        load_test_config,
        data_dir,
        work_dir
    ):
        dict_lock = work_dir.joinpath('lock')
        batch_file = work_dir.joinpath('runner', 'run_test.sh')
        command = self.config.generic.job_command
        enabled_variable_name_argumentation = self.config.generic.enabled_variable_name_argumentation
        trial_id = 99

        output_file_path = work_dir.joinpath('result', f'{trial_id}.yml')
        error_file_path = work_dir.joinpath('error', f'{trial_id}.txt')
        config_file_path = self.configs['config_abci.json']

        job_script_preamble = create_job_script_preamble(
            self.config.ABCI.job_script_preamble,
            self.config.ABCI.job_script_preamble_path
        )
        self.model.create_abci_batch_file(
            trial_id,
            self.parameters(),
            output_file_path,
            error_file_path,
            config_file_path,
            batch_file,
            job_script_preamble,
            command,
            enabled_variable_name_argumentation,
            dict_lock
        )
        assert work_dir.joinpath('runner/run_test.sh').exists()

    def parameters(self):
        return {
            'trial_id': 99,
            'parameters': [
                {
                    'parameter_name': 'x1',
                    'type': 'uniform_float',
                    'value': -4.716525234779937
                },
                {
                    'parameter_name': 'x2',
                    'type': 'uniform_float',
                    'value': 123456
                }
            ],
            'result': 74.70862563400767,
            'start_time': '11/03/2020 16:07:40',
            'end_time': '11/03/2020 16:07:40'
        }
