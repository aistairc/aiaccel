import numpy as np

from aiaccel.parameter import get_type, load_parameter
from tests.base_test import BaseTest


class TestParameter(BaseTest):

    def test_get_type(self):
        int_p = {'type': 'uniform_int'}
        assert get_type(int_p) == 'INT'
        float_p = {'type': 'uniform_float'}
        assert get_type(float_p) == 'FLOAT'
        cat_p = {'type': 'categorical'}
        assert get_type(cat_p) == 'CATEGORICAL'
        ord_p = {'type': 'ordinal'}
        assert get_type(ord_p) == 'ORDINAL'
        other_p = {'type': 'invalid'}
        assert get_type(other_p) == 'invalid'

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
        rng = np.random.RandomState(1)
        p = hp.sample(rng=rng)
        assert len(p) == 4

        # json_string['parameters'].append({'name': 'e', 'type': 'invalid'})
        json_string.append({'name': 'e', 'type': 'invalid'})
        hp = load_parameter(json_string)

        try:
            hp.sample(rng=rng)
            assert False
        except TypeError:
            assert True
