from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.random import RandomState

from aiaccel.config import Config, load_config
from aiaccel.optimizer._grid_point_generator import (
    CategoricalGridCondition, FloatGridCondition, GridCondition,
    GridConditionCollection, GridPointGenerator, GridValueType,
    IntGridCondition, NumericGridCondition, NumericType, OrdinalGridCondition,
    _cast_start_to_integer, _cast_stop_to_integer, _create_grid_condition,
    _is_there_zero_between_lower_and_upper, _validate_parameter_range)
from aiaccel.parameter import (HyperParameter, HyperParameterConfiguration,
                               load_parameter)
from tests.base_test import BaseTest


class TestGridCondition:
    @pytest.fixture(autouse=True)
    def setup(self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
        self.choices = [0, 1, 2]
        self.num_choices = len(self.choices)
        self.hyperparameter = HyperParameter(
            {
                'name': 'test',
                'type': "FLOAT",
                'lower': None,
                'upper': None,
                'log': None,
                'num_numeric_choices': None,
                'choices': None,
                'sequence': None
            }
        )
        if "noautousefixture" in request.keywords:
            yield
        else:
            with monkeypatch.context() as m:
                m.setattr(GridCondition, "__abstractmethods__", set())
                self.grid_condition = GridCondition(self.hyperparameter)
                yield
        self.choices = None
        self.num_choices = None
        self.hyperparameter = None
        self.grid_condition = None

    @pytest.mark.noautousefixture
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(TypeError):
            _ = GridCondition(self.hyperparameter)

        with monkeypatch.context() as m:
            m.setattr(GridCondition, "__abstractmethods__", set())
            grid_condition = GridCondition(self.hyperparameter)
            assert grid_condition.choices == []
            assert grid_condition.num_choices == 0

    def test_contains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition, "_choices", self.choices)
            for choice in self.grid_condition:
                assert choice in self.choices

    def test_iter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition, "_choices", self.choices)
            for i, choice in enumerate(self.grid_condition):
                assert choice == self.choices[i]

    def test_len(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition, "_num_choices", self.num_choices)
            assert len(self.grid_condition) == self.num_choices

    def test_has_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            assert self.grid_condition.has_choices() is False
            m.setattr(self.grid_condition, "_choices", self.choices)
            assert self.grid_condition.has_choices() is True

    def test_has_num_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            assert self.grid_condition.has_num_choices() is False
            m.setattr(self.grid_condition, "_num_choices", self.num_choices)
            assert self.grid_condition.has_num_choices() is True

    def test_num_choices_incrementable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            assert self.grid_condition.num_choices_incrementable() is True
            m.setattr(self.grid_condition, "_num_choices", 1)
            m.setattr(self.grid_condition, "_max_num_choices", 1)
            assert self.grid_condition.num_choices_incrementable() is False

    def test_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid_condition = GridCondition(self.hyperparameter)
        with monkeypatch.context() as m:
            m.setattr(grid_condition, "_choices", self.choices)
            assert grid_condition.choices == self.choices
        grid_condition.choices = self.choices
        assert grid_condition._choices == self.choices

    def test_num_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid_condition = GridCondition(self.hyperparameter)
        with monkeypatch.context() as m:
            m.setattr(grid_condition, "_num_choices", self.num_choices)
            assert grid_condition.num_choices == self.num_choices
        grid_condition.num_choices = self.num_choices
        assert grid_condition._num_choices == self.num_choices


argnames_is_there_zero_between_lower_and_upper = "lower, upper, expect"
argvalues_is_there_zero_between_lower_and_upper = [
    (-2, -1, False),
    (-1, 0, True),
    (-1, 1, True),
    (0, 1, True),
    (1, 2, False)
]


@pytest.mark.parametrize(
    argnames_is_there_zero_between_lower_and_upper,
    argvalues_is_there_zero_between_lower_and_upper,
)
def is_there_zero_between_lower_and_upper(lower, upper, expect) -> None:
    hyperparameter = HyperParameter(
        {
            "name": "test",
            "type": 'uniform_float',
            "lower": lower,
            "upper": upper,
            "log": None,
            "num_numeric_choices": None,
            "choices": None,
            "sequence": None
        }
    )
    assert _is_there_zero_between_lower_and_upper(hyperparameter) == expect


def test_validate_parameter_range(monkeypatch: pytest.MonkeyPatch) -> None:
    hyperparameter = HyperParameter(
        {
            "name": "test",
            "type": 'uniform_float',
            "lower": 1.0,
            "upper": 10.0,
            "log": None,
            "num_numeric_choices": None,
            "choices": None,
            "sequence": None
        }
    )
    with monkeypatch.context() as m:
        m.setattr(hyperparameter, "log", True)

        m.setattr("aiaccel.optimizer._grid_point_generator._is_there_zero_between_lower_and_upper", lambda _: True)
        with pytest.raises(ValueError):
            _validate_parameter_range(hyperparameter)

        m.setattr("aiaccel.optimizer._grid_point_generator._is_there_zero_between_lower_and_upper", lambda _: False)
        assert _validate_parameter_range(hyperparameter) is None

        m.setattr(hyperparameter, "log", False)
        assert _validate_parameter_range(hyperparameter) is None


class TestNumericGridCondition:
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'uniform_float',
                "lower": 1.0,
                "upper": 10.0,
                "log": None,
                "num_numeric_choices": None,
                "choices": None,
                "sequence": None
            }
        )

        with pytest.raises(TypeError):
            _ = NumericGridCondition(hyperparameter)

        with monkeypatch.context() as m:
            m.setattr(NumericGridCondition, "__abstractmethods__", set())
            grid_condition = NumericGridCondition(hyperparameter)
            assert grid_condition._choices == []
            assert grid_condition._num_choices == 0


argnames_float_grid_condition_create_choices = "has_num_choices, num_choices, log, num_numeric_choices, choices"
argvalues_float_grid_condition_create_choices = [
    (False, None, False, None, []),
    (False, None, False, 10, list(map(float, np.linspace(1.0, 10.0, 10)))),
    (False, None, True, None, []),
    (False, None, True, 10, list(map(float, np.geomspace(1.0, 10.0, 10)))),
    (True, 100, False, None, list(map(float, np.linspace(1.0, 10.0, 100)))),
    (True, 100, True, None, list(map(float, np.geomspace(1.0, 10.0, 100))))
]


class TestFloatGridCondition:
    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'uniform_float',
                "lower": 1.0,
                "upper": 10.0,
                "log": None,
                "num_numeric_choices": None,
                "choices": None,
                "sequence": None
            }
        )
        yield
        self.hyperparameter = None

    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            args = []
            m.setattr(GridCondition, "__init__", lambda _, hyperparameter: args.append(hyperparameter))
            _ = FloatGridCondition(self.hyperparameter)
            assert args == [self.hyperparameter]

    @pytest.mark.parametrize(
        argnames=argnames_float_grid_condition_create_choices,
        argvalues=argvalues_float_grid_condition_create_choices
    )
    def test_create_choices(
        self,
        monkeypatch: pytest.MonkeyPatch,
        has_num_choices: bool,
        num_choices: int | None,
        log: bool,
        num_numeric_choices: int | None,
        choices: list[float]
    ) -> None:
        hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'uniform_float',
                "lower": 1.0,
                "upper": 10.0,
                "log": log,
                "num_numeric_choices": num_numeric_choices,
            }
        )
        grid_condition = FloatGridCondition(hyperparameter)
        with monkeypatch.context() as m:
            m.setattr(grid_condition, "has_num_choices", lambda: has_num_choices)
            m.setattr(grid_condition, "_num_choices", num_choices)
            grid_condition.create_choices(hyperparameter)
            assert grid_condition.choices == choices

    def test_num_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'uniform_float',
                "lower": 1.0,
                "upper": 10.0,
                "log": False,
                "num_numeric_choices": None,
            }
        )
        grid_condition = FloatGridCondition(hyperparameter)
        with monkeypatch.context() as m:
            m.setattr(grid_condition, "_num_choices", 10)
            assert grid_condition.num_choices == 10
        grid_condition.num_choices = 10.1
        assert grid_condition._num_choices == 10
        grid_condition.num_choices = -1
        assert grid_condition._num_choices == 0


def test_cast_start_to_integer():
    assert _cast_start_to_integer(-1.5) == -1
    assert _cast_start_to_integer(-1.0) == -1
    assert _cast_start_to_integer(0.0) == 0
    assert _cast_start_to_integer(1.0) == 1
    assert _cast_start_to_integer(1.5) == 2


def test_cast_stop_to_integer():
    assert _cast_stop_to_integer(-1.5) == -2
    assert _cast_stop_to_integer(-1.0) == -1
    assert _cast_stop_to_integer(0.0) == 0
    assert _cast_stop_to_integer(1.0) == 1
    assert _cast_stop_to_integer(1.5) == 1


argnames_int_grid_condition_create_choices = (
    "tests_error, exception, has_num_choices, num_choices, lower, upper, log, num_numeric_choices, choices"
)
argvalues_int_grid_condition_create_choices = [
    (False, None, False, None, 1, 10, False, None, []),
    (False, None, False, None, 1, 10, False, 10, list(set(map(int, np.linspace(1, 10, 10))))),
    (False, None, False, None, -1, -3, False, 10, [-3, -2, -1]),
    (False, None, False, None, -1, 1, False, 10, [-1, 0, 1]),
    (False, None, False, None, 1, 3, False, 10, [1, 2, 3]),
    (False, None, False, None, 1.5, 3.5, False, 10, [2, 3]),
    (False, None, False, None, -1.5, -3.5, False, 10, [-3, -2]),
    (False, None, False, None, -1.5, 1.5, False, 10, [-1, 0, 1]),
    (False, None, False, None, 1, 10, True, None, []),
    (False, None, False, None, 1, 10, True, 10, list(set(map(int, np.geomspace(1, 10, 10))))),
    (False, None, True, 100, 1, 10, False, None, list(set(map(int, np.linspace(1, 10, 100))))),
    (False, None, True, 100, 1, 10, False, None, list(set(map(int, np.linspace(1, 10, 100))))),
    (True, ValueError, False, None, 1.0, 1.9, False, None, [])
]


class TestIntGridCondition:
    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.choices = [0, 1, 2]
        self.num_choices = len(self.choices)
        self.hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'uniform_int',
                "lower": 1.0,
                "upper": 10.0,
                "log": None,
                "num_numeric_choices": None,
                "choices": None,
                "sequence": None
            }
        )
        yield
        self.choices = None
        self.num_choices = None
        self.hyperparameter = None

    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            args = []
            m.setattr(GridCondition, "__init__", lambda _, hyperparameter: args.append(hyperparameter))
            _ = IntGridCondition(self.hyperparameter)
            assert args == [self.hyperparameter]

    @pytest.mark.parametrize(
        argnames=argnames_int_grid_condition_create_choices,
        argvalues=argvalues_int_grid_condition_create_choices
    )
    def test_create_choices(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tests_error: bool,
        exception: BaseException,
        has_num_choices: bool,
        num_choices: int | None,
        lower: NumericType,
        upper: NumericType,
        log: bool,
        num_numeric_choices: int | None,
        choices: list[float]
    ) -> None:
        hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'uniform_int',
                "lower": lower,
                "upper": upper,
                "log": log,
                "num_numeric_choices": num_numeric_choices,
            }
        )
        if tests_error:
            with pytest.raises(exception):
                _ = IntGridCondition(hyperparameter)
        else:
            grid_condition = IntGridCondition(hyperparameter)
            with monkeypatch.context() as m:
                m.setattr(grid_condition, "has_num_choices", lambda: has_num_choices)
                m.setattr(grid_condition, "_num_choices", num_choices)
                grid_condition.create_choices(hyperparameter)
                assert grid_condition.choices == choices

    def test_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid_condition = IntGridCondition(self.hyperparameter)
        with monkeypatch.context() as m:
            m.setattr(grid_condition, "_choices", self.choices)
            assert grid_condition.choices == self.choices
        grid_condition.choices = self.choices
        assert grid_condition._choices == self.choices
        assert grid_condition._num_choices == self.num_choices

    def test_num_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grid_condition = IntGridCondition(self.hyperparameter)
        with monkeypatch.context() as m:
            m.setattr(grid_condition, "_num_choices", self.num_choices)
            assert grid_condition.num_choices == self.num_choices
        grid_condition.num_choices = 5.1
        assert grid_condition._num_choices == 5
        grid_condition.num_choices = 100
        assert grid_condition._num_choices == grid_condition._max_num_choices
        grid_condition.num_choices = -1
        assert grid_condition._num_choices == 0


class TestCategoricalGridCondition:
    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.choices = [0, 1, 2]
        self.num_choices = len(self.choices)
        self.hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'categorical',
                "choices": self.choices,
            }
        )
        yield
        self.choices = None
        self.num_choices = None
        self.hyperparameter = None

    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            args = []
            m.setattr(GridCondition, "__init__", lambda _, hyperparameter: args.append(hyperparameter))
            _ = CategoricalGridCondition(self.hyperparameter)
            assert args == [self.hyperparameter]

    def test_create_choices(self) -> None:
        grid_condition = CategoricalGridCondition(self.hyperparameter)
        assert grid_condition.choices == self.choices
        assert grid_condition.num_choices == self.num_choices


class TestOrdinalGridCondition:
    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.choices = [0, 1, 2]
        self.num_choices = len(self.choices)
        self.hyperparameter = HyperParameter(
            {
                "name": "test",
                "type": 'ordinal',
                "sequence": self.choices,
            }
        )
        yield
        self.choices = None
        self.num_choices = None
        self.hyperparameter = None

    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            args = []
            m.setattr(GridCondition, "__init__", lambda _, hyperparameter: args.append(hyperparameter))
            _ = OrdinalGridCondition(self.hyperparameter)
            assert args == [self.hyperparameter]

    def test_create_choices(self) -> None:
        grid_condition = OrdinalGridCondition(self.hyperparameter)
        assert grid_condition.choices == self.choices
        assert grid_condition.num_choices == self.num_choices


def test_create_grid_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    hyperparameter = HyperParameter(
        {
            "name": "test",
            "type": 'uniform_float'
        }
    )
    with monkeypatch.context() as m:
        m.setattr(GridCondition, "__init__", lambda *_: None)
        hyperparameter.type = 'uniform_float'
        grid_condition = _create_grid_condition(hyperparameter)
        assert isinstance(grid_condition, FloatGridCondition)
        hyperparameter.type = 'uniform_int'
        grid_condition = _create_grid_condition(hyperparameter)
        assert isinstance(grid_condition, IntGridCondition)
        hyperparameter.type = 'categorical'
        grid_condition = _create_grid_condition(hyperparameter)
        assert isinstance(grid_condition, CategoricalGridCondition)
        hyperparameter.type = 'ordinal'
        grid_condition = _create_grid_condition(hyperparameter)
        assert isinstance(grid_condition, OrdinalGridCondition)
        with pytest.raises(TypeError):
            hyperparameter.type = "INVALID"
            _ = _create_grid_condition(hyperparameter)


argnames_grid_condition_collection_register = (
    'num_trials, parameter_type, lower, upper, num_numeric_choices, log, choices, sequence, expect'
)
argvalues_grid_condition_collection_register = [
    (10, 'uniform_float', 0.0, 1.0, 10, False, None, None, list(map(float, np.linspace(0.0, 1.0, 10)))),
    (10, 'uniform_float', 0.0, 1.0, None, False, None, None, list(map(float, np.linspace(0.0, 1.0, 10)))),
    (10, 'uniform_int', 0, 10, 10, False, None, None, list(set(np.linspace(0, 10, 10, dtype=int)))),
    (10, 'uniform_int', 0, 10, None, False, None, None, list(set(np.linspace(0, 10, 10, dtype=int)))),
    (10, 'categorical', None, None, None, None, ['a', 'b'], None, ['a', 'b']),
    (10, 'ordinal', None, None, None, None, None, [0, 1], [0, 1])
]


class TestGridConditionCollection:
    @ pytest.fixture(autouse=True)
    def setup_hyperparameters(self, request: pytest.FixtureRequest) -> Generator[None, None, None]:
        self.hyperparameters = [
            HyperParameter(
                {
                    'name': 'test',
                    'type': 'uniform_int',
                    'lower': 0.0,
                    'upper': 1.0,
                    'log': False,
                    'num_numeric_choices': None,
                }
            )
        ]
        if "noautousefixture" in request.keywords:
            yield
        else:
            self.grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
            yield
        self.hyperparameters = None
        self.grid_condition_collection = None

    @pytest.mark.noautousefixture
    def test_init(self) -> None:
        grid_condition_collection = GridConditionCollection(100, self.hyperparameters)
        assert grid_condition_collection._num_trials == 100
        assert len(grid_condition_collection._conditions) == 1

    def test_contains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conditions = [1, 2, 3]
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition_collection, "_conditions", conditions)
            for i in conditions:
                assert i in self.grid_condition_collection
            assert 4 not in self.grid_condition_collection

    def test_iter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conditions = [1, 2, 3]
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition_collection, "_conditions", conditions)
            for i, condition in enumerate(self.grid_condition_collection):
                assert condition == conditions[i]

    def test_len(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert len(self.grid_condition_collection) == 1
        conditions = [1, 2, 3]
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition_collection, "_conditions", conditions)
            assert len(self.grid_condition_collection) == len(conditions)

    @pytest.mark.parametrize(
        argnames_grid_condition_collection_register,
        argvalues_grid_condition_collection_register
    )
    def test_register_grid_conditions(
        self,
        num_trials: int,
        parameter_type: Literal['uniform_float', 'uniform_int', 'categorical', 'ordinal'],
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

    def test_get_grid_conditions_with_empty_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conditions = [
            FloatGridCondition(
                HyperParameter(
                    {
                        "name": "x1",
                        "type": "FLOAT",
                        "lower": 0.0,
                        "upper": 1.0,
                        "log": False
                    }
                )
            ),
            CategoricalGridCondition(
                HyperParameter(
                    {
                        "name": "x0",
                        "type": "CATEGORICAL",
                        "choices": [0, 1, 2]
                    }
                )
            )
        ]
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition_collection, "_conditions", conditions)
            empty_grid_conditions = self.grid_condition_collection._get_grid_conditions_with_empty_choices()
            assert empty_grid_conditions == [conditions[0]]

    def test_set_num_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conditions = [
            FloatGridCondition(
                HyperParameter(
                    {
                        "name": "x1",
                        "type": "FLOAT",
                        "lower": 0.0,
                        "upper": 1.0,
                        "log": False
                    }
                )
            )
        ]
        with monkeypatch.context() as m:
            m.setattr(self.grid_condition_collection, "_conditions", conditions)
            self.grid_condition_collection._set_num_choices(conditions)
            assert conditions[0].num_choices == 100

    def test_choices(self) -> None:
        pass


argnames_grid_point_generator = 'sampling_method, rng'
argvalues_grid_point_generator = [
    ('IN_ORDER', None),
    ('UNIFORM', None),
    ('RANDOM', RandomState(42)),
    ('DUPLICATABLE_RANDOM', RandomState(42)),
    ('INVALID', None)
]


class TestGridPointGenerator(BaseTest):
    @ pytest.fixture(autouse=True)
    def setup_grid_point_generator(
        self,
        data_dir: Path,
        create_tmp_config: Callable[[Path], Path]
    ) -> Generator[None, None, None]:
        self.data_dir = data_dir

        config = self.load_config_for_test(self.configs['config_budget-specified-grid.json'])
        self.hyperparameters = load_parameter(config.optimize.parameters).get_parameter_list()
        self.trial_number = config.optimize.trial_number
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

    @ pytest.mark.parametrize(argnames_grid_point_generator, argvalues_grid_point_generator)
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
