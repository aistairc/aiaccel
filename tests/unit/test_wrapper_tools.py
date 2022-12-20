from aiaccel.util.filesystem import create_yaml
from aiaccel.util.time_tools import get_time_now
from aiaccel.wrapper_tools import create_runner_command, save_result

from tests.base_test import BaseTest


class TestCeaterRunnerComand(BaseTest):

    def test_create_runner_command(
        self,
        clean_work_dir,
        get_one_parameter,
        load_test_config,
        work_dir
    ):

        for d in self.test_result_data:
            name = f"{d['trial_id']}.yml"
            path = work_dir / 'result' / name
            create_yaml(path, d)

        config = load_test_config()
        dict_lock = work_dir.joinpath('lock')
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
        }

        commands = create_runner_command(
            config.generic.job_command,
            get_one_parameter(),
            'name',
            'config.json'
        )
        assert commands[0] == 'python'
        assert commands[1] == 'original_main.py'
        # assert commands[2] == '-i'
        assert commands[2] == '--trial_id'
        assert commands[3] == 'name'
        # assert commands[4] == '-c'
        assert commands[4] == '--config'
        assert commands[5] == 'config.json'
        # assert commands[6] == '-x1=0.9932890709584586'
        assert commands[6] == '--x1=0.9932890709584586'

        start_time = get_time_now()
        end_time = get_time_now()
        assert save_result(
            work_dir,
            dict_lock,
            'name',
            {},
            start_time,
            end_time
        ) is None

        assert save_result(
            work_dir,
            dict_lock,
            'name',
            {},
            start_time,
            end_time,
            err_message='error'
        ) is None
