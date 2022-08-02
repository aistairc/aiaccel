from aiaccel.optimizer.sobol.search import Optimizer
from tests.base_test import BaseTest
import pytest


class TestOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'optimizer'
        }
        self.optimizer = Optimizer(options)
        yield
        self.optimizer = None

    def test_pre_process(self):
        assert self.optimizer.pre_process() is None

    def test_generate_parameter(self):
        self.optimizer.pre_process()
        assert self.optimizer.generate_parameter() is None
