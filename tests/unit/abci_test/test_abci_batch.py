from aiaccel.abci import create_abci_batch_file
from tests.base_test import BaseTest


class TestCreateAbciBatchFile(BaseTest):

    def test_create_abci_batch_file(
        self,
        clean_work_dir,
        load_test_config,
        data_dir,
        work_dir
    ):

        config = load_test_config()
        dict_lock = work_dir.joinpath('lock')
        batch_file = work_dir.joinpath('runner', 'run_test.sh')
        command = config.job_command.get()
        trial_id = 99

        error_file_path = work_dir.joinpath('error', f'{trial_id}.txt')
        workspace_path = config.workspace.get()
        config_file_path = self.config_json

        job_script_preamble = data_dir.joinpath(config.job_script_preamble.get())
        create_abci_batch_file(
            trial_id,
            self.parameters(),
            workspace_path,
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
                    'type': 'FLOAT',
                    'value': -4.716525234779937
                },
                {
                    'parameter_name': 'x2',
                    'type': 'FLOAT',
                    'value': 123456
                }
            ],
            'result': 74.70862563400767,
            'start_time': '11/03/2020 16:07:40',
            'end_time': '11/03/2020 16:07:40'
        }
