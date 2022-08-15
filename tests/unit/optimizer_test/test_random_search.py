from aiaccel.optimizer.random.search import RandomSearchOptimizer
from tests.base_test import BaseTest


class TestRandomSearchOptimizer(BaseTest):

    def test_generate_parameter(self):
        options = {
            'config': str(self.config_random_path),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        optimizer = RandomSearchOptimizer(options)
        optimizer.storage.alive.init_alive()
        optimizer.pre_process()
        assert optimizer.generate_parameter() is None
