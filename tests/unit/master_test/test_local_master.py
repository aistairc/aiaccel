from aiaccel.master import LocalMaster

from tests.base_test import BaseTest


class TestLocalMaster(BaseTest):

    def test_init(self, clean_work_dir):
        master = LocalMaster(self.load_config_for_test(self.configs["config.json"]))
        assert master.loop_start_time is None
