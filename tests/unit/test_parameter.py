import numpy as np
import pytest

from aiaccel.parameter import HyperParameterConfiguration
from aiaccel.cli import get_best_parameter
from aiaccel.common import dict_lock, dict_result, goal_maximize, goal_minimize
from aiaccel.util import create_yaml
from tests.base_test import BaseTest


class TestParameter(BaseTest):

    def test_get_best_parameters(
        self,
        work_dir,
        clean_work_dir
    ):
        clean_work_dir()

        objective_y_index = 0
        files = list(work_dir.joinpath(dict_result).glob('*.yml'))
        best, best_file = get_best_parameter(
            files,
            goal_maximize,
            objective_y_index,
            work_dir.joinpath(dict_lock)
        )
        assert best is None
        assert best_file is None

        results = [120, 101., np.float64(140.)]

        for i in range(len(self.test_result_data)):
            d = self.test_result_data[i]
            name = f"{d['trial_id']}.yml"
            path = work_dir / 'result' / name
            d['result'] = [results[i]]
            create_yaml(path, d)

        files = list(work_dir.joinpath(dict_result).glob('*.yml'))
        files.sort()
        best, best_file = get_best_parameter(
            files,
            goal_maximize,
            objective_y_index,
            work_dir.joinpath(dict_lock)
        )
        assert best == 140.
        best, best_file = get_best_parameter(
            files,
            goal_minimize,
            objective_y_index,
            work_dir.joinpath(dict_lock)
        )
        assert best == 101.
        try:
            _, _ = get_best_parameter(
                files,
                'invalid_goal',
                objective_y_index,
                work_dir.joinpath(dict_lock)
            )
            assert False
        except ValueError:
            assert True

    def test_get_parameter_list(self):
        hp = HyperParameterConfiguration(self.load_config_for_test(self.configs["config.json"]).optimize.parameters)
        p = hp.get_parameter_list()
        assert len(p) == 10

    def test_get_hyperparameter(self):
        hp = HyperParameterConfiguration(self.load_config_for_test(self.configs["config.json"]).optimize.parameters)
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
        hp = HyperParameterConfiguration(json_string)

        with pytest.raises(TypeError):
            hp.sample()

        with pytest.raises(TypeError):
            hp.sample(initial=True)

        rng = np.random.RandomState(1)
        p = hp.sample(rng)
        assert len(p) == 4

        json_string.append({'name': 'e', 'type': 'invalid'})
        hp = HyperParameterConfiguration(json_string)

        try:
            hp.sample(rng=rng)
            assert False
        except TypeError:
            assert True
