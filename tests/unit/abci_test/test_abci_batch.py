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
        error_output = str(work_dir / 'error' / f"{d['trial_id']}.txt")

        commands = create_runner_command(
            config.job_command.get(),
            get_one_parameter(),
            'test',
            'config.json',
            error_output
        )
        wrapper_file = data_dir.joinpath(config.job_script_preamble.get())
        create_abci_batch_file(batch_file, wrapper_file, commands, dict_lock)
        assert work_dir.joinpath('runner/run_test.sh').exists()
