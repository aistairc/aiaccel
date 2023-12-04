from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.common import goal_maximize
from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import NelderMead, NelderMeadOptimizer
from aiaccel.parameter import HyperParameterConfiguration
from tests.base_test import BaseTest


class TestNelderMeadOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir):
        self.optimizer = NelderMeadOptimizer(self.load_config_for_test(self.configs["config.json"]))
        yield
        self.optimizer = None

    def test_generate_initial_parameter(self):
        expected = [
            {'parameter_name': 'x1', 'type': 'uniform_float', 'value': 0.74, 'out_of_boundary': False},
            {'parameter_name': 'x2', 'type': 'uniform_float', 'value': 2.98, 'out_of_boundary': False},
            {'parameter_name': 'x3', 'type': 'uniform_float', 'value': 3.62, 'out_of_boundary': False},
            {'parameter_name': 'x4', 'type': 'uniform_float', 'value': 0.9, 'out_of_boundary': False},
            {'parameter_name': 'x5', 'type': 'uniform_float', 'value': 1.99, 'out_of_boundary': False},
            {'parameter_name': 'x6', 'type': 'uniform_float', 'value': -2.78, 'out_of_boundary': False},
            {'parameter_name': 'x7', 'type': 'uniform_float', 'value': 1.0, 'out_of_boundary': False},
            {'parameter_name': 'x8', 'type': 'uniform_float', 'value': 4.97, 'out_of_boundary': False},
            {'parameter_name': 'x9', 'type': 'uniform_float', 'value': 1.98, 'out_of_boundary': False},
            {'parameter_name': 'x10', 'type': 'uniform_float', 'value': 4.03, 'out_of_boundary': False}
        ]

        _optimizer = NelderMeadOptimizer(self.load_config_for_test(self.configs["config.json"]))
        _optimizer._rng = np.random.RandomState(0)
        _nelder_mead = _optimizer.generate_initial_parameter()
        self.optimizer._rng = np.random.RandomState(0)

        with patch.object(self.optimizer, "nelder_mead", None):
            initial_params = self.optimizer.generate_initial_parameter()
            for i in range(len(expected)):
                assert initial_params[i] == expected[i]

        with patch.object(self.optimizer, "nelder_mead", _nelder_mead):
            assert self.optimizer.generate_initial_parameter() is None

    def test_generate_parameter(
        self,
        load_test_config_org,
        setup_result
    ):
        config = self.load_config_for_test(self.configs["config_nelder_mead.json"])
        self.optimizer = NelderMeadOptimizer(config)
        self.optimizer.params = ConvertedParameterConfiguration(
            HyperParameterConfiguration(config.optimize.parameters), convert_log=True, convert_int=True,
            convert_choices=True, convert_sequence=True,
        )
        self.optimizer.generate_initial_parameter()
        assert len(self.optimizer.generate_parameter()) > 0

        self.optimizer = NelderMeadOptimizer(self.load_config_for_test(self.configs["config.json"]))
        self.optimizer.generate_initial_parameter()
        with patch.object(self.optimizer, 'nelder_mead_main', return_value=[]):
            with patch.object(self.optimizer, 'single_or_multiple_trial_params', []):
                assert self.optimizer.generate_parameter() is None

    def test_nelder_mead_main(self):
        config = self.load_config_for_test(self.configs["config_nelder_mead.json"])
        self.optimizer = NelderMeadOptimizer(config)
        self.optimizer.generate_initial_parameter()
        self.optimizer.nelder_mead_main()

        with patch.object(self.optimizer.nelder_mead, 'search', return_value=[]):
            assert self.optimizer.nelder_mead_main() == []
