from aiaccel.scheduler.local_scheduler import LocalScheduler
from tests.base_test import BaseTest


class TestLocalScheduler(BaseTest):

    def test_get_stats(self, clean_work_dir, config_json, fake_process):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'scheduler'
        }
        scheduler = LocalScheduler(options)
        fake_process.register_subprocess(
            ['/bin/ps', '-eo', 'pid,user,stat,lstart,args'],
            stdout=[
                "PID ARGS                          USER          STAT "
                "STARTED\n"
                # "1   python wrapper.py -i sample1  root          Ss   Mon Oct "
                "1   python wrapper.py --index sample1  root          Ss   Mon Oct "
                "10 00:00:00 2020\n"
                # "2   python wrapper.py -i sample2  root          Ss   Mon Oct "
                "2   python wrapper.py --index sample2  root          Ss   Mon Oct "
                "10 00:00:10 2020\n"
            ]
        )
        scheduler.jobs.append({'hashname': 'sample1', 'thread': None})
        assert scheduler.get_stats() is None
