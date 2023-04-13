from aiaccel.abci import create_abci_batch_file
from aiaccel.util import create_yaml
from aiaccel.wrapper_tools import create_runner_command
from tests.base_test import BaseTest


class TestCreateAbciBatchFile(BaseTest):

    def test_create_abci_batch_file(
        self,
        clean_work_dir,
        get_one_parameter,
        load_test_config,
        data_dir,
        work_dir
    ):

        for d in self.test_result_data:
            name = f"{d['trial_id']}.yml"
            path = work_dir / 'result' / name
            create_yaml(path, d)

        config = load_test_config()
        dict_lock = work_dir.joinpath('lock')
        batch_file = work_dir.joinpath('runner', 'run_test.sh')
        command = config.generic.job_command
        trial_id = 99

        output_file_path = work_dir.joinpath('result', f'{trial_id}.yml')
        error_file_path = work_dir.joinpath('error', f'{trial_id}.txt')
        config_file_path = self.configs['config.json']

        job_script_preamble = data_dir.joinpath(config.ABCI.job_script_preamble)
        create_abci_batch_file(
            trial_id,
            self.parameters(),
            output_file_path,
            error_file_path,
            config_file_path,
            batch_file,
            job_script_preamble,
            command,
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
