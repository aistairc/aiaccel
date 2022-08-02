from aiaccel.optimizer.random.search import Optimizer
from tests.base_test import BaseTest


class TestOptimizer(BaseTest):

    def test_generate_parameter(self):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'optimizer'
        }
        optimizer = Optimizer(options)
        optimizer.pre_process()
        assert optimizer.generate_parameter() is None
