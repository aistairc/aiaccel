from aiaccel.util.filesystem import create_yaml
from aiaccel.util.time_tools import get_time_now
from aiaccel.wrapper_tools import create_runner_command
from tests.base_test import BaseTest


class TestCeaterRunnerComand(BaseTest):

    def test_create_runner_command(
        self,
        clean_work_dir,
        work_dir,
        load_test_config
    ):
        clean_work_dir()

        config = load_test_config()
        d = self.test_result_data[0]
        error_output = str(work_dir / 'error' / f"{d['trial_id']}.txt")

        commands = create_runner_command(
            config.job_command.get(),
            d,
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
