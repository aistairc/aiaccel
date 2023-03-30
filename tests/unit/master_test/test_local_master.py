from aiaccel.command_line_options import CommandLineOptions
from aiaccel.master import LocalMaster

from tests.base_test import BaseTest


class TestLocalMaster(BaseTest):

    def test_init(self, clean_work_dir):
        options = CommandLineOptions(
            config=str(self.config_json),
            resume=None,
            clean=False,
            process_name="master"
        )
        master = LocalMaster(options)
        assert master.loop_start_time is None
