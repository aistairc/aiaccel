from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.random import RandomState

from aiaccel.config import Config
from aiaccel.optimizer._grid_point_generator import _make_numeric_choices
from aiaccel.optimizer._grid_point_generator import GridCondition
from aiaccel.optimizer._grid_point_generator import GridConditionCollection
from aiaccel.optimizer._grid_point_generator import GridPointGenerator
from aiaccel.optimizer._grid_point_generator import GridValueType
from aiaccel.parameter import HyperParameter
from aiaccel.parameter import load_parameter

from tests.base_test import BaseTest


condition_key1 = 'parameter_type, lower, upper, num_numeric_choices, log, num_choices, expect'
parameter_conditions1 = [
    ('FLOAT', 0.0, 1.0, 10, False, None, np.linspace(0.0, 1.0, 10).tolist()),
    ('FLOAT', 0.0, 1.0, None, False, 20, np.linspace(0.0, 1.0, 20).tolist()),
    ('FLOAT', 0.1, 1.0, 10, True, None, np.geomspace(0.1, 1.0, 10).tolist()),
    ('FLOAT', 0.1, 1.0, None, True, 20, np.geomspace(0.1, 1.0, 20).tolist()),
    ('INT', 0, 10, 10, False, None, np.linspace(0, 10, 10, dtype=int).tolist()),
    ('INT', 0, 10, None, False, 20, np.linspace(0, 10, 20, dtype=int).tolist()),
    ('INT', 1, 10, 10, True, None, np.geomspace(1, 10, 10, dtype=int).tolist()),
    ('INT', 1, 10, None, True, 20, np.geomspace(1, 10, 20, dtype=int).tolist()),
]

condition_key2 = 'parameter_type, lower, upper, num_numeric_choices, log, choices, sequence, expect'
parameter_conditions2 = [
    ('FLOAT', 0.0, 1.0, 10, False, None, None, np.linspace(0.0, 1.0, 10).tolist()),
    ('FLOAT', 0.0, 1.0, None, False, None, None, []),
    ('INT', 0, 10, 10, False, None, None, np.linspace(0, 10, 10, dtype=int).tolist()),
    ('INT', 0, 10, None, False, None, None, []),
    ('CATEGORICAL', None, None, None, None, ['a', 'b'], None, ['a', 'b']),
    ('ORDINAL', None, None, None, None, None, [0, 1], [0, 1])
]

condition_key3 = 'num_trials, parameter_type, lower, upper, num_numeric_choices, log, choices, sequence, expect'
parameter_conditions3 = [
    (10, 'FLOAT', 0.0, 1.0, 10, False, None, None, np.linspace(0.0, 1.0, 10).tolist()),
    (10, 'FLOAT', 0.0, 1.0, None, False, None, None, np.linspace(0.0, 1.0, 10).tolist()),
    (10, 'INT', 0, 10, 10, False, None, None, np.linspace(0, 10, 10, dtype=int).tolist()),
    (10, 'INT', 0, 10, None, False, None, None, np.linspace(0, 10, 10, dtype=int).tolist()),
    (10, 'CATEGORICAL', None, None, None, None, ['a', 'b'], None, ['a', 'b']),
    (10, 'ORDINAL', None, None, None, None, None, [0, 1], [0, 1])
]

sampling_condition_key = 'sampling_method, rng'
sampling_methods = [
    ('IN_ORDER', None),
    ('UNIFORM', None),
    ('RANDOM', RandomState(42)),
    ('DUPLICATABLE_RANDOM', RandomState(42)),
    ('INVALID', None)
]


@ pytest.mark.parametrize(condition_key1, parameter_conditions1)
def test_make_numeric_choices(
    parameter_type: Literal['FLOAT', 'INT'],
    lower: float | int,
    upper: float | int,
    num_numeric_choices: int,
    log: bool,
    num_choices: int | None,
    expect: list[GridValueType]
) -> None:
    hyperparameter = HyperParameter(
        {
            'name': 'test',
            'type': parameter_type,
            'lower': lower,
            'upper': upper,
            'log': log,
            'num_numeric_choices': num_numeric_choices
        }
    )
    choices = _make_numeric_choices(hyperparameter, num_choices)
    assert choices == expect

    hyperparameter.type = 'INVALID'
    with pytest.raises(TypeError):
        _ = _make_numeric_choices(hyperparameter, num_choices)


class TestGridCondition(BaseTest):
    @pytest.mark.parametrize(condition_key2, parameter_conditions2)
    def test_init(
        self,
        parameter_type: Literal['FLOAT', 'INT', 'CATEGORICAL', 'ORDINAL'],
        lower: float | int | None,
        upper: float | int | None,
        num_numeric_choices: int | None,
        log: bool | None,
        choices: list[GridValueType] | None,
        sequence: list[GridValueType] | None,
        expect: list[GridValueType]
    ) -> None:
        hyperparameter = HyperParameter(
            {
                'name': 'test',
                'type': parameter_type,
                'lower': lower,
                'upper': upper,
                'log': log,
                'num_numeric_choices': num_numeric_choices,
                'choices': choices,
                'sequence': sequence
            }
        )
        grid_condition = GridCondition(hyperparameter)
        assert grid_condition.choices == expect

        hyperparameter.type = 'INVALID'
        with pytest.raises(TypeError):
            _ = GridCondition(hyperparameter)

    def test_iter(self) -> None:
        choices = ["a", "b", "c"]
        hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": "CATEGORICAL",
                "choices": choices
            }
        )
        grid_condition = GridCondition(hyperparameter)
        for i, item in enumerate(grid_condition):
            assert item == choices[i]

    def test_len(self) -> None:
        choices = ["1", "2", "3"]
        hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": "CATEGORICAL",
                "choices": choices
            }
        )
        grid_condition = GridCondition(hyperparameter)
        assert len(grid_condition) == len(choices)


class TestGridConditionCollection(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_hyperparameters(self):
        self.hyperparameters = [
            HyperParameter(
                {
                    'name': 'test',
                    'type': 'FLOAT',
                    'lower': 0.0,
                    'upper': 1.0,
                    'log': False,
                    'num_numeric_choices': None,
                }
            )
        ]

    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
        assert grid_condition_collection.num_trials == 100
        assert len(grid_condition_collection._conditions) == 1

        with monkeypatch.context() as m:
            m.setattr(GridConditionCollection, '_register_grid_conditions', lambda *_: [1])
            with pytest.raises(ValueError):
                _ = GridConditionCollection(100, self.hyperparameters)

    def test_iter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
        conditions = [1, 2, 3]
        with monkeypatch.context() as m:
            m.setattr(grid_condition_collection, '_conditions', conditions)
            for i, condition in enumerate(grid_condition_collection):
                assert condition == conditions[i]

    @pytest.mark.parametrize(condition_key3, parameter_conditions3)
    def test_register_grid_conditions(
        self,
        num_trials: int,
        parameter_type: Literal['FLOAT', 'INT', 'CATEGORICAL', 'ORDINAL'],
        lower: float | int | None,
        upper: float | int | None,
        num_numeric_choices: int | None,
        log: bool | None,
        choices: list[GridValueType] | None,
        sequence: list[GridValueType] | None,
        expect: list[GridValueType]
    ) -> None:
        hyperparameters = [
            HyperParameter(
                {
                    'name': 'test',
                    'type': parameter_type,
                    'lower': lower,
                    'upper': upper,
                    'log': log,
                    'num_numeric_choices': num_numeric_choices,
                    'choices': choices,
                    'sequence': sequence
                }
            )
        ]
        grid_condition_collection = GridConditionCollection(num_trials, hyperparameters)
        assert grid_condition_collection.choices[0] == expect

    def test_update_least_space_size(self) -> None:
        grid_condition = GridCondition(self.hyperparameters[0])
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)

        grid_condition_collection._least_space_size = 1
        grid_condition.num_choices = 2
        grid_condition_collection._update_least_space_size(grid_condition)
        assert grid_condition_collection._least_space_size == 2

        grid_condition_collection._least_space_size = 1
        grid_condition.num_choices = 0
        grid_condition_collection._update_least_space_size(grid_condition)
        assert grid_condition_collection._least_space_size == 1

    def test_update_unspecified_parameters(self) -> None:
        grid_condition = GridCondition(self.hyperparameters[0])
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)

        grid_condition_collection._num_unspecified_parameters = 0
        grid_condition.num_choices = 1
        grid_condition_collection._num_larger_choice_parameters = 0
        grid_condition_collection._update_num_unspecified_parameters(grid_condition)
        assert grid_condition_collection._num_unspecified_parameters == 0
        assert grid_condition_collection._num_larger_choice_parameters == 0

        grid_condition.num_choices = 0
        grid_condition_collection._update_num_unspecified_parameters(grid_condition)
        assert grid_condition_collection._num_unspecified_parameters == 1
        assert grid_condition_collection._num_larger_choice_parameters == 1

    def test_update_residual_space_size(self) -> None:
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
        grid_condition_collection._residual_space_size = 100
        grid_condition_collection._least_space_size = 10
        grid_condition_collection._update_residual_space_size()
        assert grid_condition_collection._residual_space_size == 10

    def test_calc_num_choices(self) -> None:
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
        grid_condition_collection._num_larger_choices = 1
        grid_condition_collection._num_smaller_choices = 0
        grid_condition_collection._num_unspecified_parameters = 2
        grid_condition_collection._residual_space_size = 10.0
        grid_condition_collection._calc_num_choices()
        assert grid_condition_collection._num_larger_choices == 4
        assert grid_condition_collection._num_smaller_choices == 3

    def test_split_num_unspecified_parameters(self) -> None:
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
        grid_condition_collection._num_larger_choices = 3
        grid_condition_collection._num_smaller_choices = 2
        grid_condition_collection._num_larger_choice_parameters = 5
        grid_condition_collection._num_smaller_choice_parameters = 0
        grid_condition_collection._residual_space_size = 100.0
        grid_condition_collection._split_num_unspecified_parameters()
        assert grid_condition_collection._num_larger_choice_parameters == 3
        assert grid_condition_collection._num_smaller_choice_parameters == 2

    def test_get_auto_defined_num_choices(self) -> None:
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
        grid_condition_collection._num_larger_choices = 3
        grid_condition_collection._num_smaller_choices = 2
        grid_condition_collection._num_larger_choice_parameters = 3
        grid_condition_collection._num_smaller_choice_parameters = 2
        nums_choices = grid_condition_collection._get_auto_defined_num_choices()
        assert nums_choices == [3, 3, 3, 2, 2]


class TestGridPointGenerator(BaseTest):
    @ pytest.fixture(autouse=True)
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
            self.trial_number, self.hyperparameters.copy()
        )
        yield
        self.grid_point_generator = None

    def test_init(self) -> None:
        with pytest.raises(ValueError):
            _ = GridPointGenerator(
                num_trials=0,
                hyperparameters=self.hyperparameters.copy(),
                accept_small_trial_number=False
            )
        grid_point_generator = GridPointGenerator(
            num_trials=0,
            hyperparameters=self.hyperparameters,
            accept_small_trial_number=True
        )
        assert len(grid_point_generator._grid_condition_collection) > 0

        grid_point_generator = GridPointGenerator(
            num_trials=self.trial_number,
            hyperparameters=self.hyperparameters,
            sampling_method='IN_ORDER',
            rng=None,
            accept_small_trial_number=False
        )
        assert len(grid_point_generator._grid_point_stack) == 0

        grid_point_generator = GridPointGenerator(
            num_trials=self.trial_number,
            hyperparameters=self.hyperparameters,
            sampling_method='RANDOM',
            rng=None,
            accept_small_trial_number=False
        )
        assert len(grid_point_generator._grid_point_stack) > 0

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

    @ pytest.mark.parametrize(sampling_condition_key, sampling_methods)
    def test_get_next_grid_point(
        self,
        sampling_method: Literal['IN_ORDER', 'UNIFORM', 'RANDOM', 'DUPLICATABLE_RANDOM', 'INVALID'],
        rng: RandomState,
    ) -> None:
        grid_point_generator = GridPointGenerator(
            self.trial_number,
            self.hyperparameters,
            sampling_method=sampling_method,
            rng=rng
        )
        try:
            next_grid_point = grid_point_generator.get_next_grid_point()
            assert len(next_grid_point) > 0
        except BaseException as e:
            with pytest.raises(ValueError):
                raise e

    def test_get_grid_point_in_order(self) -> None:
        next_grid_point = self.grid_point_generator._get_grid_point_in_order(0)
        assert len(next_grid_point) > 0

    def test_get_grid_point_uniformly(self) -> None:
        next_grid_point = self.grid_point_generator._get_grid_point_uniformly(0)
        assert len(next_grid_point) > 0

    def test_get_grid_point_randomly(self) -> None:
        grid_point_generator = GridPointGenerator(
            num_trials=self.trial_number,
            hyperparameters=self.hyperparameters,
            sampling_method='RANDOM',
            rng=None
        )
        next_grid_point = grid_point_generator._get_grid_point_randomly(0)
        assert len(next_grid_point) > 0

    def test_get_grid_point_duplicatable_randomly(self) -> None:
        grid_point_generator = GridPointGenerator(
            num_trials=self.trial_number,
            hyperparameters=self.hyperparameters,
            sampling_method='DUPLICATABLE_RANDOM',
            rng=None
        )
        next_grid_point = (
            grid_point_generator._get_grid_point_duplicatable_randomly(0)
        )
        assert len(next_grid_point) > 0

    def test_num_generated_points(self) -> None:
        assert self.grid_point_generator.num_generated_points == 0
        _ = self.grid_point_generator.get_next_grid_point()
        assert self.grid_point_generator.num_generated_points == 1
