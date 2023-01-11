from aiaccel.optimizer.random_optimizer import RandomOptimizer

from tests.base_test import BaseTest


class TestRandomOptimizer(BaseTest):

<<<<<<< HEAD
    def test_generate_parameter(self):
        optimizer = RandomOptimizer(self.configs['config_random.json'])
=======
    def test_generate_parameter(self, create_tmp_config):
        self.config_random_path = create_tmp_config(self.config_random_path)
        options = {
            'config': str(self.config_random_path),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        optimizer = RandomOptimizer(options)
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
        optimizer.pre_process()
        assert len(optimizer.generate_parameter()) > 0
