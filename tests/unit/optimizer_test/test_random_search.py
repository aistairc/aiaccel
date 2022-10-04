from aiaccel.optimizer.random.search import RandomOptimizer
from tests.base_test import BaseTest


class TestRandomOptimizer(BaseTest):

    def test_generate_parameter(self):
        options = {
            'config': str(self.config_random_path),
            'resume': None,
            'clean': False,
            'fs': False,
            'module_name': 'optimizer'
        }
        optimizer = RandomOptimizer(options)
        optimizer.storage.alive.init_alive()
        optimizer.pre_process()
        assert optimizer.generate_parameter() is None
