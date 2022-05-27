from aiaccel.master.local_master import LocalMaster
from tests.base_test import BaseTest


class TestLocalMaster(BaseTest):

    def test_init(self, clean_work_dir):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'master'
        }
        master = LocalMaster(options)
        assert master.loop_start_time is None
