import unittest
from unittest.mock import MagicMock
from aiaccel.manager.pylocal_manager import PylocalManager
from aiaccel.optimizer import create_optimizer
from tests.base_test import BaseTest


class TestPylocalManager(BaseTest):

    def test__init__(self):
        config = self.load_config_for_test(self.configs['config_pylocal.json'])
        optimizer = create_optimizer(config.optimize.search_algorithm)(config)
        manager = PylocalManager(config, optimizer)
        assert manager.__init__(config, optimizer) is None

    def test_inner_loop_main_process(self, setup_hp_ready):
        config = self.load_config_for_test(self.configs['config_pylocal.json'])
        with self.create_main():
            optimizer = create_optimizer(config.optimize.search_algorithm)(config)
            manager = PylocalManager(config, optimizer)
            manager.pre_process()
            setup_hp_ready(1)
            try:
                assert manager.inner_loop_main_process() is True
            except Exception as e:
                print(e)
                assert False

    def test_get_any_trial_xs(self, setup_hp_ready):
        config = self.load_config_for_test(self.configs['config_pylocal.json'])
        with self.create_main():
            optimizer = create_optimizer(config.optimize.search_algorithm)(config)
            manager = PylocalManager(config, optimizer)
            manager.pre_process()
            setup_hp_ready(1)
            manager.inner_loop_main_process()
            xs = manager.get_any_trial_xs(1)
            assert xs == {'x1': 1.69, 'x2': 2.27, 'x3': 4.38, 'x4': 2.0, 'x5': 3.9, 'x6': 4.62, 'x7': -2.2, 'x8': 4.77, 'x9': -3.66, 'x10': 3.59}
