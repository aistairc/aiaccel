import numpy as np
import pytest

from aiaccel.parameter import HyperParameterConfiguration
from tests.base_test import BaseTest


class TestParameter(BaseTest):

    def test_get_parameter_list(self):
        hp = HyperParameterConfiguration(self.load_config_for_test(self.configs["config.json"]).optimize.parameters)
        p = hp.get_parameter_list()
        assert len(p) == 10

    def test_get_hyperparameter(self):
        hp = HyperParameterConfiguration(self.load_config_for_test(self.configs["config.json"]).optimize.parameters)
        ps = hp.get_hyperparameter('x3')
        assert ps.name == 'x3'

    def test_sample(self):
        json_string = [
            {
                'name': 'a',
                'type': 'uniform_int',
                'lower': 0,
                'upper': 10
            },
            {
                'name': 'b',
                'type': 'uniform_float',
                'lower': 0.,
                'upper': 10.
            },
            {
                'name': 'c',
                'type': 'categorical',
                'choices': ['red', 'green', 'blue']
            },
            {
                'name': 'd',
                'type': 'ordinal',
                'sequence': ['10', '20', '30']
            }
        ]
        hp = HyperParameterConfiguration(json_string)

        with pytest.raises(TypeError):
            hp.sample()

        with pytest.raises(TypeError):
            hp.sample(initial=True)

        rng = np.random.RandomState(1)
        p = hp.sample(rng)
        assert len(p) == 4

        json_string.append({'name': 'e', 'type': 'invalid'})
        try:
            hp = HyperParameterConfiguration(json_string)
            assert False
        except TypeError:
            assert True

    def test_initial(self):
        # If initial is out of range, raises AssertionError
        json_string = [
            {
                'name': 'a',
                'type': 'uniform_int',
                'lower': 0,
                'upper': 10,
                'initial': -1
            }
        ]
        try:
            HyperParameterConfiguration(json_string)
            assert False
        except AssertionError:
            assert True

        json_string = [
            {
                'name': 'b',
                'type': 'uniform_float',
                'lower': 0.,
                'upper': 10.,
                'initial': -0.1
            }
        ]
        try:
            HyperParameterConfiguration(json_string)
            assert False
        except AssertionError:
            assert True

        json_string = [
            {
                'name': 'c',
                'type': 'categorical',
                'choices': ['red', 'green', 'blue'],
                'initial': 'yellow'
            }
        ]
        try:
            HyperParameterConfiguration(json_string)
            assert False
        except AssertionError:
            assert True

        json_string = [
            {
                'name': 'd',
                'type': 'ordinal',
                'sequence': ['10', '20', '30'],
                'initial': '40'
            }
        ]
        try:
            HyperParameterConfiguration(json_string)
            assert False
        except AssertionError:
            assert True
