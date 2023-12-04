from aiaccel.optimizer.random_optimizer import RandomOptimizer

from tests.base_test import BaseTest


class TestRandomOptimizer(BaseTest):

    def test_generate_parameter(self):
        optimizer = RandomOptimizer(self.load_config_for_test(self.configs['config_random.json']))
        assert len(optimizer.generate_parameter()) > 0
