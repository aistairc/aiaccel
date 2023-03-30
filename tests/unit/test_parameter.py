import numpy as np
import pytest

from aiaccel.parameter import load_parameter
from tests.base_test import BaseTest


class TestParameter(BaseTest):

    def test_load_parameter(self):
        hp = load_parameter(
            # self.config.get('hyperparameter', 'ConfigSpace_hyperparameter'))
            # self.config.get('optimize', 'parameters')
            self.config.hyperparameters.get()
        )
        assert hp.__class__.__name__ == 'HyperParameterConfiguration'

    def test_get_parameter_list(self):
        hp = load_parameter(
            # self.config.get('hyperparameter', 'ConfigSpace_hyperparameter'))
            # self.config.get('optimize', 'parameters')
            self.config.hyperparameters.get()

        )
        p = hp.get_parameter_list()
        assert len(p) == 10

    def test_get_hyperparameter(self):
        hp = load_parameter(
            # self.config.get('hyperparameter', 'ConfigSpace_hyperparameter'))
            # self.config.get('optimize', 'parameters')
            self.config.hyperparameters.get()
        )
        ps = hp.get_hyperparameter('x3')
        assert ps.name == 'x3'

    def test_sample(self):
        # json_string = {
        #     'parameters': [
        #         {'name': 'a', 'type': 'uniform_int', 'lower': 0, 'upper': 10},
        #         {'name': 'b', 'type': 'uniform_float', 'lower': 0.,
        #          'upper': 10.},
        #         {'name': 'c', 'type': 'categorical',
        #          'choices': ['red', 'green', 'blue']},
        #         {'name': 'd', 'type': 'ordinal',
        #          'sequence': ['10', '20', '30']}
        #     ]
        # }
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
        hp = load_parameter(json_string)

        with pytest.raises(TypeError):
            hp.sample()

        with pytest.raises(TypeError):
            hp.sample(initial=True)

        rng = np.random.RandomState(1)
        p = hp.sample(rng)
        assert len(p) == 4

        json_string.append({'name': 'e', 'type': 'invalid'})
        hp_with_invalid_type = load_parameter(json_string)
        with pytest.raises(TypeError):
            hp_with_invalid_type.sample(rng)
