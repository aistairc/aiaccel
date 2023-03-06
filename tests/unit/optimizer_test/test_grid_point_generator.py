from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aiaccel.optimizer._grid_point_generator import _count_fixed_grid_points
from aiaccel.optimizer._grid_point_generator import _generate_all_grid_points
from aiaccel.optimizer._grid_point_generator import _suggest_nums_grid_points
from aiaccel.optimizer._grid_point_generator import GridPointGenerator

from tests.base_test import BaseTest


@dataclass
class HyperParameter:
    name: str
    type: str
    lower: float | int | None = None
    upper: float | int | None = None
    log: bool | None = None
    num_grid_points: int | None = None
    choices: list[Any] | None = None
    sequence: list[Any] | None = None


def test_count_fixed_grid_points():

    hyperparameters = [
        HyperParameter(name='x1', type='FLOAT', lower=-1.0, upper=1.0, log=False),
        HyperParameter(name='x2', type='FLOAT', lower=-1.0, upper=1.0, log=False),
        HyperParameter(name='x3', type='FLOAT', lower=-1.0, upper=1.0, log=False),
    ]
    assert _count_fixed_grid_points(hyperparameters) == [0, 0, 0]

    hyperparameters = [
        HyperParameter(name='x1', type='FLOAT', lower=-1.0, upper=1.0, log=False, num_grid_points=0),
        HyperParameter(name='x2', type='FLOAT', lower=-1.0, upper=1.0, log=False, num_grid_points=1),
        HyperParameter(name='x3', type='FLOAT', lower=-1.0, upper=1.0, log=False, num_grid_points=2),
    ]
    assert _count_fixed_grid_points(hyperparameters) == [0, 1, 2]

    hyperparameters = [
        HyperParameter(name='x1', type='CATEGORICAL', choices=['a', 'b', 'c']),
        HyperParameter(name='x2', type='ORDINAL', sequence=[1, 2, 4]),
        HyperParameter(name='x3', type='INT', lower=-1, upper=1, log=False, num_grid_points=10),
    ]
    assert _count_fixed_grid_points(hyperparameters) == [3, 3, 10]


def test_suggest_nums_grid_points():
    grid_space_size = 10 ** 3
    least_grid_space_size = 10 ** 2
    num_parameter = 1
    assert _suggest_nums_grid_points(grid_space_size, least_grid_space_size, num_parameter) == [10]

    num_parameter = 2
    assert _suggest_nums_grid_points(grid_space_size, least_grid_space_size, num_parameter) == [3, 4]

    num_parameter = 3
    assert _suggest_nums_grid_points(grid_space_size, least_grid_space_size, num_parameter) == [2, 2, 3]

    num_parameter = 4
    assert _suggest_nums_grid_points(grid_space_size, least_grid_space_size, num_parameter) == [2, 2, 2, 2]

    num_parameter = 5
    assert _suggest_nums_grid_points(grid_space_size, least_grid_space_size, num_parameter) == [1, 2, 2, 2, 2]


def test_generate_all_grid_points():
    num_trials = 9
    hyperparameters = [
        HyperParameter(name='x1', type='FLOAT', lower=0.0, upper=2.0),
        HyperParameter(name='x2', type='INT', lower=0, upper=2)
    ]
    num_fixed_hyperparameters = _count_fixed_grid_points(hyperparameters)
    least_grid_space_size = np.prod(num_fixed_hyperparameters, where=(np.array(num_fixed_hyperparameters) != 0))
    nums_grid_points = _suggest_nums_grid_points(num_trials, least_grid_space_size, num_fixed_hyperparameters.count(0))
    assert _generate_all_grid_points(hyperparameters, nums_grid_points) == [[0.0, 1.0, 2.0], [0, 1, 2]]

    num_trials = 9
    hyperparameters = [
        HyperParameter(name='x1', type='FLOAT', lower=0.0, upper=2.0),
        HyperParameter(name='x2', type='INT', lower=0, upper=2, num_grid_points=4)
    ]
    num_fixed_hyperparameters = _count_fixed_grid_points(hyperparameters)
    least_grid_space_size = np.prod(num_fixed_hyperparameters, where=(np.array(num_fixed_hyperparameters) != 0))
    nums_grid_points = _suggest_nums_grid_points(num_trials, least_grid_space_size, num_fixed_hyperparameters.count(0))
    assert _generate_all_grid_points(hyperparameters, nums_grid_points) == [[0.0, 1.0, 2.0], [0, 0, 1, 2]]

    num_trials = 9
    hyperparameters = [
        HyperParameter(name='x1', type='FLOAT', lower=0.0, upper=2.0),
        HyperParameter(name='x2', type='INT', lower=0, upper=3, num_grid_points=4)
    ]
    num_fixed_hyperparameters = _count_fixed_grid_points(hyperparameters)
    least_grid_space_size = np.prod(num_fixed_hyperparameters, where=(np.array(num_fixed_hyperparameters) != 0))
    nums_grid_points = _suggest_nums_grid_points(num_trials, least_grid_space_size, num_fixed_hyperparameters.count(0))
    assert _generate_all_grid_points(hyperparameters, nums_grid_points) == [[0.0, 1.0, 2.0], [0, 1, 2, 3]]

    num_trials = 0
    hyperparameters = [
        HyperParameter(name='x1', type='FLOAT', lower=0.0, upper=2.0),
        HyperParameter(name='x2', type='INT', lower=0, upper=3, num_grid_points=4)
    ]
    num_fixed_hyperparameters = _count_fixed_grid_points(hyperparameters)
    least_grid_space_size = np.prod(num_fixed_hyperparameters, where=(np.array(num_fixed_hyperparameters) != 0))
    nums_grid_points = _suggest_nums_grid_points(num_trials, least_grid_space_size, num_fixed_hyperparameters.count(0))
    assert _generate_all_grid_points(hyperparameters, nums_grid_points) == [[0.0], [0, 1, 2, 3]]

    num_trials = 0
    hyperparameters = [
        HyperParameter(name='x1', type='CATEGORICAL', choices=['a', 'b', 'c']),
        HyperParameter(name='x2', type='ORDINAL', sequence=[1, 2])
    ]
    num_fixed_hyperparameters = _count_fixed_grid_points(hyperparameters)
    least_grid_space_size = np.prod(num_fixed_hyperparameters, where=(np.array(num_fixed_hyperparameters) != 0))
    nums_grid_points = _suggest_nums_grid_points(num_trials, least_grid_space_size, num_fixed_hyperparameters.count(0))
    assert _generate_all_grid_points(hyperparameters, nums_grid_points) == [['a', 'b', 'c'], [1, 2]]


class TestGridPointGenerator(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_optimizer(self, data_dir, create_tmp_config):
        self.data_dir = data_dir
        self.config_grid = create_tmp_config(self.config_grid)
