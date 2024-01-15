from aiaccel.manager import AbciManager
from aiaccel.optimizer import create_optimizer

from tests.base_test import BaseTest


class TestAbciManager(BaseTest):

    def test_parse_trial_id(
        self,
        config_json,
        database_remove
    ):
        database_remove()
        config = self.load_config_for_test(self.configs['config.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = AbciManager(config, optimizer)
        s = {"name": "run_000005.sh"}
        trial_id = int(manager.parse_trial_id(s['name']))
        assert trial_id == 5

        s = {"name": "run_xxxxxx.sh"}
        trial_id = manager.parse_trial_id(s['name'])
        assert trial_id is None
