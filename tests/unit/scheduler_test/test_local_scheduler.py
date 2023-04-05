from aiaccel.scheduler import Job
from aiaccel.scheduler import LocalScheduler

from tests.base_test import BaseTest


class TestLocalScheduler(BaseTest):

    def test_get_stats(self, clean_work_dir, config_json, fake_process):
        config = self.load_config_for_test(self.configs['config.json'])
        scheduler = LocalScheduler(config)
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
        job = Job(config, scheduler, scheduler.create_model(), trial_id)
        scheduler.jobs.append(job)
        assert scheduler.get_stats() is None

    def test_parse_trial_id(self, config_json, database_remove):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        scheduler = LocalScheduler(config)
        s = {"name": "2 python user.py --trial_id 5 --config config.yaml --x1=1.0 --x2=1.0"}
        trial_id = int(scheduler.parse_trial_id(s['name']))
        assert trial_id == 5
