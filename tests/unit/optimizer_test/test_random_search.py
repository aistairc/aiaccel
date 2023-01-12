from aiaccel.optimizer.random_optimizer import RandomOptimizer

from tests.base_test import BaseTest


class TestRandomOptimizer(BaseTest):

    def test_generate_parameter(self):
        optimizer = RandomOptimizer(self.configs['config_random.json'])
        optimizer.pre_process()
        assert len(optimizer.generate_parameter()) > 0
