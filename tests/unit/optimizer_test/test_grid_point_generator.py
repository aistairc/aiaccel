from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.random import RandomState

from aiaccel.config import Config
from aiaccel.optimizer._grid_point_generator import _count_fixed_grid_points
from aiaccel.optimizer._grid_point_generator import _generate_all_grid_points
from aiaccel.optimizer._grid_point_generator import _suggest_nums_grid_points
from aiaccel.optimizer._grid_point_generator import GridPointGenerator
from aiaccel.parameter import load_parameter

from tests.base_test import BaseTest


condition_key = 'parameter_type, num_grid_point, choices, sequence, expect'
parameter_conditions = [
    ('FLOAT', None, None, None, [0]),
    ('FLOAT', 10, None, None, [10]),
    ('INT', None, None, None, [0]),
    ('INT', 10, None, None, [10]),
    ('CATEGORICAL', None, ['a', 'b'], None, [2]),
    ('ORDINAL', None, None, [1, 2], [2]),
    ('INVALID', None, None, None, [])
]

sampling_condition_key = 'sampling_method, rng'
sampling_methods = [
    ('IN_ORDER', None),
    ('UUNIFORM', None),
    ('RANDOM', RandomState(42)),
    ('DUPLICATABLE_RANDOM', RandomState(42))
]


@pytest.mark.parametrize(condition_key, parameter_conditions)
def test_count_fixed_grid_points(
    data_dir: Path,
    create_tmp_config: Callable[[Path], Path],
    parameter_type: Literal['FLOAT', 'INT', 'CATEGORICAL', 'ORDINAL', 'INVALID'],
    num_grid_point: int | None,
    choices: list[str] | None,
    sequence: list[int] | None,
    expect: list[int]
) -> None:
    config_path = create_tmp_config(
        data_dir / 'config_budget-specified-grid.json'
    )
    config = Config(config_path)
    params = config.hyperparameters.get()
    hyperparameters = load_parameter(params).get_parameter_list()

    for hyperparameter in hyperparameters:
        hyperparameter.type = parameter_type
        hyperparameter.num_grid_points = num_grid_point
        hyperparameter.choices = choices
        hyperparameter.sequence = sequence
    nums_grid_points = _count_fixed_grid_points(hyperparameters)
    assert nums_grid_points == expect * len(hyperparameters)


def test_suggest_nums_grid_points() -> None:
    grid_space_size = 10 ** 3
    least_grid_space_size = 10 ** 2
    num_parameter = 1
    nums_grid_points = _suggest_nums_grid_points(
        grid_space_size,
        least_grid_space_size,
        num_parameter
    )
    assert nums_grid_points == [10]

    nums_grid_points = _suggest_nums_grid_points(
        grid_space_size,
        least_grid_space_size,
        num_parameter=0
    )
    assert nums_grid_points == []


@pytest.mark.parametrize(condition_key, parameter_conditions)
def test_generate_all_grid_points(
    data_dir: Path,
    create_tmp_config: Callable[[Path], Path],
    parameter_type: Literal['FLOAT', 'INT', 'CATEGORICAL', 'ORDINAL', 'INVALID'],
    num_grid_point: int | None,
    choices: list[str] | None,
    sequence: list[int] | None,
    expect: list[int]
) -> None:
    config_path = create_tmp_config(
        data_dir / 'config_budget-specified-grid.json'
    )
    config = Config(config_path)
    trial_number = config.trial_number.get()
    params = config.hyperparameters.get()

    hyperparameters = load_parameter(params).get_parameter_list()

    for hyperparameter in hyperparameters:
        hyperparameter.type = parameter_type
        hyperparameter.num_grid_points = num_grid_point
        hyperparameter.choices = choices
        hyperparameter.sequence = sequence

    num_fixed_hyperparameters = _count_fixed_grid_points(hyperparameters)
    least_grid_space_size = np.prod(
        num_fixed_hyperparameters,
        where=(np.array(num_fixed_hyperparameters) != 0)
    )
    nums_grid_points = _suggest_nums_grid_points(
        trial_number,
        least_grid_space_size,
        num_fixed_hyperparameters.count(0)
    )

    all_grid_points = _generate_all_grid_points(hyperparameters, nums_grid_points)
    assert len(all_grid_points) == len(expect) * len(hyperparameters)


class TestGridPointGenerator(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_grid_point_generator(
        self,
        data_dir: Path,
        create_tmp_config: Callable[[Path], Path]
    ) -> Generator[None, None, None]:
        self.data_dir = data_dir
        self.config_path = create_tmp_config(
            self.data_dir / 'config_budget-specified-grid.json'
        )

        config = Config(self.config_path)
        params = config.hyperparameters.get()
        self.hyperparameters = load_parameter(params).get_parameter_list()
        self.trial_number = config.trial_number.get()
        self.grid_point_generator = GridPointGenerator(
            self.hyperparameters, self.trial_number
        )
        yield
        self.grid_point_generator = None

    def test_init(self) -> None:
        with pytest.raises(ValueError):
            _ = GridPointGenerator(
                hyperparameters=self.hyperparameters,
                trial_number=0,
                accept_small_trial_number=False
            )

        grid_point_generator = GridPointGenerator(
            hyperparameters=self.hyperparameters,
            trial_number=0,
            accept_small_trial_number=True
        )
        assert len(grid_point_generator._point_list) > 0

        grid_point_generator = GridPointGenerator(
            hyperparameters=self.hyperparameters,
            trial_number=self.trial_number,
            sampling_method='IN_ORDER',
            rng=None,
            accept_small_trial_number=False
        )
        assert len(grid_point_generator._grid_point_stack) == 0
        assert len(grid_point_generator._generated_grid_point_stack) == 0

        grid_point_generator = GridPointGenerator(
            hyperparameters=self.hyperparameters,
            trial_number=self.trial_number,
            sampling_method='RANDOM',
            rng=None,
            accept_small_trial_number=False
        )
        assert len(grid_point_generator._grid_point_stack) > 0
        assert len(grid_point_generator._generated_grid_point_stack) == 0

    def test_all_grid_points_generated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self.grid_point_generator.all_grid_points_generated() is False

        with monkeypatch.context() as m:
            m.setattr(
                self.grid_point_generator,
                '_num_generated_points',
                self.grid_point_generator._grid_space_size
            )
            assert self.grid_point_generator.all_grid_points_generated()

    @pytest.mark.parametrize(sampling_condition_key, sampling_methods)
    def test_get_next_grid_point(
        self,
        sampling_method: Literal['IN_ORDER', 'UUNIFORM', 'RANDOM', 'DUPLICATABLE_RANDOM'],
        rng: RandomState
    ) -> None:
        grid_point_generator = GridPointGenerator(
            self.hyperparameters,
            self.trial_number,
            sampling_method=sampling_method,
            rng=rng
        )
        next_grid_point = grid_point_generator.get_next_grid_point()
        assert len(next_grid_point) > 0

    def test_get_grid_point_in_order(self) -> None:
        next_grid_point = self.grid_point_generator._get_grid_point_in_order(0)
        assert len(next_grid_point) > 0

    def test_get_grid_point_thin_out(self) -> None:
        next_grid_point = self.grid_point_generator._get_grid_point_thin_out(0)
        assert len(next_grid_point) > 0

    def test_get_grid_point_random(self) -> None:
        grid_point_generator = GridPointGenerator(
            hyperparameters=self.hyperparameters,
            trial_number=self.trial_number,
            sampling_method='RANDOM',
            rng=None
        )
        next_grid_point = grid_point_generator._get_grid_point_random(0)
        assert len(next_grid_point) > 0

    def test_get_grid_point_duplicatable_random(self) -> None:
        grid_point_generator = GridPointGenerator(
            hyperparameters=self.hyperparameters,
            trial_number=self.trial_number,
            sampling_method='DUPLICATABLE_RANDOM',
            rng=None
        )
        next_grid_point = (
            grid_point_generator._get_grid_point_duplicatable_random(0)
        )
        assert len(next_grid_point) > 0

    def test_num_generated_points(self) -> None:
        assert self.grid_point_generator.num_generated_points == 0
        _ = self.grid_point_generator.get_next_grid_point()
        assert self.grid_point_generator.num_generated_points == 1
