from datetime import datetime

from aiaccel.common import datetime_format
from aiaccel.util import create_yaml
from aiaccel.wrapper_tools import create_runner_command, save_result
from tests.base_test import BaseTest


class TestCeaterRunnerComand(BaseTest):

    def test_create_runner_command(
        self,
        clean_work_dir,
        work_dir,
        get_one_parameter,
        load_test_config
    ):
        clean_work_dir()

        for d in self.test_result_data:
            name = f"{d['trial_id']}.yml"
            path = work_dir / 'result' / name
            create_yaml(path, d)

        config = load_test_config()
        dict_lock = work_dir.joinpath('lock')
        error_output = str(work_dir / 'error' / f"{d['trial_id']}.txt")

        commands = create_runner_command(
            config.generic.job_command,
            get_one_parameter(),
            'name',
            'config.json',
            error_output
        )
        print(commands)
        assert commands[0] == 'python'
        assert commands[1] == 'original_main.py'
        assert commands[2] == '--x1'
        assert commands[3] == '0.9932890709584586'
        # skip --x2 ~ x10
        assert commands[22] == '--trial_id'
        assert commands[23] == 'name'
        assert commands[24] == '--config'
        assert commands[25] == 'config.json'
        assert commands[26] == '2>'
        assert commands[27] == error_output

        start_time = datetime.now().strftime(datetime_format)
        end_time = datetime.now().strftime(datetime_format)
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
