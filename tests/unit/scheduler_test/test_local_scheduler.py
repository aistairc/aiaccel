from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.job.job import Job
from aiaccel.config import load_config

from tests.base_test import BaseTest


class TestLocalScheduler(BaseTest):

<<<<<<< HEAD
    def test_get_stats(self, clean_work_dir, config_json, fake_process):
        config = load_config(config_json)
        scheduler = LocalScheduler(config)
=======
    def test_get_stats(self, fake_process):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        scheduler = LocalScheduler(options)
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
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

        trial_id = 1
        job = Job(self.config, scheduler, trial_id)
        scheduler.jobs.append(job)
        assert scheduler.get_stats() is None

<<<<<<< HEAD
    def test_parse_trial_id(self, config_json):
        config = load_config(config_json)
        scheduler = LocalScheduler(config)
=======
    def test_parse_trial_id(self):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'scheduler'
        }

        scheduler = LocalScheduler(options)
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
        s = {"name": "2 python user.py --trial_id 5 --config config.yaml --x1=1.0 --x2=1.0"}
        trial_id = int(scheduler.parse_trial_id(s['name']))
        assert trial_id == 5
