from aiaccel.scheduler.abci_scheduler import AbciScheduler
from tests.base_test import BaseTest


class TestAbciScheduler(BaseTest):

    def test_get_stats(
        self,
        clean_work_dir,
        config_json,
        data_dir,
        fake_process
    ):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = AbciScheduler(options)
        xml_path = data_dir.joinpath('qstat.xml')
        fake_process.register_subprocess(['qstat', '-xml'], stdout=[])
        assert scheduler.get_stats() is None

        with open(xml_path, 'r') as f:
            xml_string = f.read()

        fake_process.register_subprocess(
            ['qstat', '-xml'],
            stdout=[xml_string]
        )
        assert scheduler.get_stats() is None
