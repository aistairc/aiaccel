from aiaccel.scheduler.local import LocalScheduler
from tests.base_test import BaseTest


class TestLocalScheduler(BaseTest):

    def test_get_stats(self, clean_work_dir, config_json, fake_process):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = LocalScheduler(options)
        fake_process.register_subprocess(
            ['/bin/ps', '-eo', 'pid,user,stat,lstart,args'],
            stdout=[
                "PID ARGS                          USER          STAT "
                "STARTED\n"
                # "1   python wrapper.py -i sample1  root          Ss   Mon Oct "
                "1   python wrapper.py --trial_id 1  root          Ss   Mon Oct "
                "10 00:00:00 2020\n"
                # "2   python wrapper.py -i sample2  root          Ss   Mon Oct "
                "2   python wrapper.py --trial_id 2  root          Ss   Mon Oct "
                "10 00:00:10 2020\n"
            ]
        )
        scheduler.jobs.append({'trial_id': 'sample1', 'thread': None})
        assert scheduler.get_stats() is None

    def test_parse_trial_id(self, config_json):
        options = {
            'config': config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'fs': False,
            'process_name': 'scheduler'
        }

        scheduler = LocalScheduler(options)
        s = {"name": "2 python user.py --trial_id 5 --config config.yaml --x1=1.0 --x2=1.0"}
        trial_id = int(scheduler.parse_trial_id(s['name']))
        assert trial_id == 5
