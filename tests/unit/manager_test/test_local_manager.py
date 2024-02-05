from aiaccel.manager import Job, LocalManager
from aiaccel.optimizer import create_optimizer

from tests.base_test import BaseTest


class TestLocalManager(BaseTest):

    def test_parse_trial_id(self, config_json, database_remove):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = LocalManager(config, optimizer)
        s = {"name": "2 python user.py --trial_id=5 --config=config.yaml --x1=1.0 --x2=1.0"}
        trial_id = int(manager.parse_trial_id(s['name']))
        assert trial_id == 5
