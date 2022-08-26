from aiaccel.master.local import LocalMaster
from tests.base_test import BaseTest


class TestLocalMaster(BaseTest):

    def test_init(self, clean_work_dir):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        master = LocalMaster(options)
        assert master.loop_start_time is None
