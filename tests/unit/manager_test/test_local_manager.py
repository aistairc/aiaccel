from aiaccel.manager import Job, LocalManager
from aiaccel.optimizer import create_optimizer

from tests.base_test import BaseTest


class TestLocalManager(BaseTest):

    def test_get_stats(self, clean_work_dir, config_json, fake_process):
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
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
        job = Job(config, manager, manager.create_model(), trial_id)
        manager.jobs.append(job)
        assert manager.get_stats() is None

    def test_parse_trial_id(self, config_json, database_remove):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        s = {"name": "2 python user.py --trial_id=5 --config=config.yaml --x1=1.0 --x2=1.0"}
        trial_id = int(manager.parse_trial_id(s['name']))
        assert trial_id == 5
