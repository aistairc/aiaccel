from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from numpy.random import RandomState

from aiaccel.converted_parameter import (
    ConvertedCategoricalParameter,
    ConvertedFloatParameter,
    ConvertedIntParameter,
    ConvertedNumericalParameter,
    ConvertedOrdinalParameter,
    ConvertedParameter,
    ConvertedParameterConfiguration,
    WeightOfChoice,
    _convert_float,
    _convert_int,
    _convert_numerics,
    _decode_weight_distribution,
    _is_weight_collected,
    _make_structured_value,
    _make_weight_distribution,
    _make_weight_name,
    _restore_float,
    _restore_int,
)
from aiaccel.parameter import HyperParameter, HyperParameterConfiguration

float_parameter_dict = {
    "name": "x0",
    "type": "uniform_float",
    "lower": 1.0,
    "upper": 2.0,
    "log": False,
    "initial": 1.0,
}

int_parameter_dict = {
    "name": "x1",
    "type": "uniform_int",
    "lower": 1,
    "upper": 2,
    "log": False,
    "initial": 1,
}

categorical_parameter_dict = {
    "name": "x2",
    "type": "categorical",
    "choices": ["a", "b", "c"],
    "initial": "a",
}

ordinal_parameter_dict = {
    "name": "x3",
    "type": "ordinal",
    "sequence": [1, 2, 3],
    "initial": 1,
}


class BaseTestConvertedParameter:
    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.float_param = HyperParameter(float_parameter_dict)
        self.int_param = HyperParameter(int_parameter_dict)
        self.categorical_param = HyperParameter(categorical_parameter_dict)
        self.ordinal_param = HyperParameter(ordinal_parameter_dict)

        self._rng = RandomState(42)

        yield

        self.float_param = None
        self.int_param = None
        self.categorical_param = None
        self.ordinal_param = None


class TestConvertedParameter(BaseTestConvertedParameter):
    def test_init(self) -> None:
        param = ConvertedParameter(self.float_param)
        assert param.name == self.float_param.name
        assert param.type == self.float_param.type
        assert param.original_initial == self.float_param.initial


class TestConvertedNumericalParameter(BaseTestConvertedParameter):
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        param = ConvertedNumericalParameter(self.float_param, convert_log=True)
        assert param.convert_log is False
        assert param.original_lower == self.float_param.lower
        assert param.original_upper == self.float_param.upper

        with monkeypatch.context() as m:
            m.setattr(self.float_param, "log", True)
            param = ConvertedNumericalParameter(self.float_param, convert_log=True)
            assert param.convert_log is True

        with monkeypatch.context() as m:
            m.setattr(self.float_param, "initial", None)
            param = ConvertedNumericalParameter(self.float_param, convert_log=True)
            assert param.original_initial is None


class TestConvertedFloatParameter(BaseTestConvertedParameter):
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        param = ConvertedFloatParameter(self.float_param)
        assert param.lower == self.float_param.lower
        assert param.upper == self.float_param.upper
        assert param.initial == self.float_param.initial

        with monkeypatch.context() as m:
            m.setattr(self.float_param, "initial", None)
            param = ConvertedFloatParameter(self.float_param)
            assert param.initial is None

        with monkeypatch.context() as m:
            m.setattr(self.float_param, "log", True)
            param = ConvertedFloatParameter(self.float_param)
            assert param.convert_log is True
            assert param.lower == np.log(self.float_param.lower)
            assert param.upper == np.log(self.float_param.upper)
            assert param.initial == np.log(self.float_param.initial)

    def test_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        param = ConvertedFloatParameter(self.float_param)
        sampled_value = param.sample(self._rng)
        assert sampled_value["name"] == self.float_param.name
        assert sampled_value["type"] == self.float_param.type
        assert self.float_param.lower <= sampled_value["value"] < self.float_param.upper

        initial_value = param.sample(self._rng, initial=True)
        assert initial_value["value"] == self.float_param.initial

        with monkeypatch.context() as m:
            m.setattr(self.float_param, "initial", None)
            param = ConvertedFloatParameter(self.float_param)
            sampled_value = param.sample(self._rng, initial=True)
            assert sampled_value["value"] is not None


class TestConvertedIntParameter(BaseTestConvertedParameter):
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        param = ConvertedIntParameter(self.int_param)
        assert param.convert_int is True
        assert param.lower == float(self.int_param.lower)
        assert param.upper == float(self.int_param.upper)
        assert param.initial == float(self.int_param.initial)

        with monkeypatch.context() as m:
            m.setattr(self.int_param, "initial", None)
            param = ConvertedIntParameter(self.int_param)
            assert param.initial is None

        with monkeypatch.context() as m:
            m.setattr(self.int_param, "log", True)
            param = ConvertedIntParameter(self.int_param)
            assert param.lower == float(np.log(self.int_param.lower))
            assert param.upper == float(np.log(self.int_param.upper))
            assert param.initial == float(np.log(self.int_param.initial))

        with monkeypatch.context() as m:
            m.setattr(self.int_param, "log", True)
            param = ConvertedIntParameter(self.int_param, convert_int=False)
            assert param.lower == int(np.log(self.int_param.lower))
            assert param.upper == int(np.log(self.int_param.upper))
            assert param.initial == int(np.log(self.int_param.initial))

    def test_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        param = ConvertedIntParameter(self.int_param)
        sampled_value = param.sample(self._rng)
        assert sampled_value["name"] == self.int_param.name
        assert sampled_value["type"] == self.int_param.type
        assert self.int_param.lower <= sampled_value["value"] < self.int_param.upper

        initial_value = param.sample(self._rng, initial=True)
        assert initial_value["value"] == float(self.int_param.initial)

        param = ConvertedIntParameter(self.int_param, convert_int=False)
        sampled_value = param.sample(self._rng)
        assert isinstance(sampled_value["value"], int)

        with monkeypatch.context() as m:
            m.setattr(self.int_param, "initial", None)
            param = ConvertedFloatParameter(self.int_param)
            sampled_value = param.sample(self._rng, initial=True)
            assert sampled_value["value"] is not None


class TestConvertedCategoricalParameter(BaseTestConvertedParameter):
    def test_init(self) -> None:
        param = ConvertedCategoricalParameter(self.categorical_param)
        assert param.choices == self.categorical_param.choices
        assert param.convert_choices is True

    def test_smaple(self) -> None:
        param = ConvertedCategoricalParameter(self.categorical_param)
        sampled_value = param.sample(self._rng)
        assert sampled_value["name"] == self.categorical_param.name
        assert sampled_value["type"] == self.categorical_param.type
        assert sampled_value["value"] in self.categorical_param.choices

        initial_value = param.sample(self._rng, initial=True)
        assert initial_value["value"] == self.categorical_param.initial


class TestConvertedOrdinalParameter(BaseTestConvertedParameter):
    def test_init(self) -> None:
        param = ConvertedOrdinalParameter(self.ordinal_param)
        assert param.sequence == self.ordinal_param.sequence
        assert param.convert_sequence is True

    def test_smaple(self) -> None:
        param = ConvertedOrdinalParameter(self.ordinal_param)
        sampled_value = param.sample(self._rng)
        assert sampled_value["name"] == self.ordinal_param.name
        assert sampled_value["type"] == self.ordinal_param.type
        assert sampled_value["value"] in self.ordinal_param.sequence

        initial_value = param.sample(self._rng, initial=True)
        assert initial_value["value"] == self.ordinal_param.initial


class TestWeightOfChoice(BaseTestConvertedParameter):
    def test_init(self) -> None:
        choice_index = 0
        param = WeightOfChoice(self.categorical_param, choice_index)
        assert param.choice_index == choice_index
        assert param.original_name == self.categorical_param.name
        assert param.name == f"{self.categorical_param.name}_{choice_index}"
        assert param.lower == 0.0
        assert param.upper == 1.0

    def test_sample(self) -> None:
        choice_index = 0
        param = WeightOfChoice(self.categorical_param, choice_index)
        sampled_value = param.sample(self._rng)
        assert sampled_value["name"] == param.name
        assert sampled_value["type"] == param.type
        assert 0.0 <= sampled_value["value"] < 1.0

        initial_value = param.sample(self._rng, initial=True)
        assert initial_value["value"] == 1.0

        choice_index = 1
        param = WeightOfChoice(self.categorical_param, choice_index)
        initial_value = param.sample(self._rng, initial=True)
        assert initial_value["value"] == 0.0


class TestConvertedHyperparameterConfiguraion:
    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.list_config = [
            float_parameter_dict, int_parameter_dict, categorical_parameter_dict, ordinal_parameter_dict
        ]
        self.params = HyperParameterConfiguration(self.list_config)
        self.converted_params = ConvertedParameterConfiguration(self.params)
        self._rng = RandomState(42)
        yield
        self.list_config = None
        self.params = None
        self.converted_params = None
        self._rng = None

    def test_init(self) -> None:
        params = ConvertedParameterConfiguration(
            self.params, convert_log=False, convert_int=False, convert_choices=False, convert_sequence=False
        )
        assert len(params._converted_params) == len(self.list_config)

    def test_convert(self, caplog: pytest.LogCaptureFixture) -> None:
        params = ConvertedParameterConfiguration(
            self.params, convert_log=True, convert_int=True, convert_choices=False, convert_sequence=False
        )
        assert len(caplog.records) == 0

        caplog.clear()

        params = ConvertedParameterConfiguration(
            self.params, convert_log=True, convert_int=True, convert_choices=True, convert_sequence=True
        )
        assert len(caplog.records) == 2
        assert "converted" in caplog.records[-1].message

        assert (
            len(params._converted_params)
            == 1 + 1 + len(categorical_parameter_dict["choices"]) + len(ordinal_parameter_dict["sequence"])
        )

        for param in params._converted_params.values():
            if isinstance(param, ConvertedFloatParameter):
                assert param.convert_log is False
            elif isinstance(param, ConvertedIntParameter):
                assert param.convert_log is False
                assert param.convert_int is True
            elif isinstance(param, WeightOfChoice):
                assert isinstance(param.initial, float)
            elif isinstance(param, ConvertedOrdinalParameter):
                assert False
            elif isinstance(param, ConvertedCategoricalParameter):
                assert False
            else:
                assert False, f"{type(param)}"

    def test_to_original_repr(self) -> None:
        params_in_internal_repr = [
            {
                "parameter_name": float_parameter_dict["name"],
                "type": float_parameter_dict["type"],
                "value": float_parameter_dict["lower"],
            },
            {
                "parameter_name": int_parameter_dict["name"],
                "type": int_parameter_dict["type"],
                "value": int_parameter_dict["lower"],
            },
            {
                "parameter_name": f"{categorical_parameter_dict['name']}_0",
                "type": categorical_parameter_dict["type"],
                "value": 0.0,
            },
            {
                "parameter_name": f"{categorical_parameter_dict['name']}_1",
                "type": categorical_parameter_dict["type"],
                "value": 0.0,
            },
            {
                "parameter_name": f"{categorical_parameter_dict['name']}_2",
                "type": categorical_parameter_dict["type"],
                "value": 1.0,
            },
            {
                "parameter_name": f"{ordinal_parameter_dict['name']}_0",
                "type": ordinal_parameter_dict["type"],
                "value": 0.0,
            },
            {
                "parameter_name": f"{ordinal_parameter_dict['name']}_1",
                "type": ordinal_parameter_dict["type"],
                "value": 1.0,
            },
            {
                "parameter_name": f"{ordinal_parameter_dict['name']}_2",
                "type": ordinal_parameter_dict["type"],
                "value": 0.0,
            },
        ]

        params_in_original_repr = self.converted_params.to_original_repr(params_in_internal_repr)
        assert len(params_in_original_repr) == len(self.list_config)  # == 4

        self._rng.shuffle(params_in_internal_repr)
        params_in_original_repr = self.converted_params.to_original_repr(params_in_internal_repr)
        assert len(params_in_original_repr) == len(self.list_config)  # == 4
        for param in params_in_original_repr:
            if param["type"].lower() == "categorical":
                assert param["value"] == categorical_parameter_dict["choices"][2]
            if param["type"].lower() == "ordinal":
                assert param["value"] == ordinal_parameter_dict["sequence"][1]

        params = HyperParameterConfiguration(self.list_config)
        converted_params = ConvertedParameterConfiguration(params, convert_choices=False, convert_sequence=False)

        params_in_internal_repr = [
            {
                "parameter_name": float_parameter_dict["name"],
                "type": float_parameter_dict["type"],
                "value": float_parameter_dict["lower"],
            },
            {
                "parameter_name": int_parameter_dict["name"],
                "type": int_parameter_dict["type"],
                "value": int_parameter_dict["lower"],
            },
            {
                "parameter_name": f"{categorical_parameter_dict['name']}",
                "type": categorical_parameter_dict["type"],
                "value": categorical_parameter_dict["choices"][2],
            },
            {
                "parameter_name": f"{ordinal_parameter_dict['name']}",
                "type": ordinal_parameter_dict["type"],
                "value": ordinal_parameter_dict["sequence"][1],
            },
        ]

        params_in_original_repr = converted_params.to_original_repr(params_in_internal_repr)
        assert len(params_in_original_repr) == len(self.list_config)  # == 4

    def test_sample(self) -> None:
        sampled_values = self.converted_params.sample(self._rng)
        assert (
            len(sampled_values)
            == 1 + 1 + len(categorical_parameter_dict["choices"]) + len(ordinal_parameter_dict["sequence"])
        )

    def test_get_hyperparameter(self) -> None:
        param = self.converted_params.get_hyperparameter(float_parameter_dict["name"])
        assert isinstance(param, ConvertedFloatParameter)
        assert param.name == float_parameter_dict["name"]

        with pytest.raises(KeyError):
            self.converted_params.get_hyperparameter("invalid_parameter_name")

    def test_get_parameter_list(self) -> None:
        parameter_list = self.converted_params.get_parameter_list()
        assert isinstance(parameter_list, list)
        assert (
            len(parameter_list)
            == 1 + 1 + len(categorical_parameter_dict["choices"]) + len(ordinal_parameter_dict["sequence"])
        )

    def test_get_parameter_dict(self) -> None:
        parameter_dict = self.converted_params.get_parameter_dict()
        assert isinstance(parameter_dict, dict)
        assert (
            len(parameter_dict)
            == 1 + 1 + len(categorical_parameter_dict["choices"]) + len(ordinal_parameter_dict["sequence"])
        )


def test_make_weight_name() -> None:
    original_name = "x"
    choice_index = 0
    name = _make_weight_name(original_name, choice_index)
    assert name == "x_0"


def test_convert_numerics(monkeypatch: pytest.MonkeyPatch) -> None:
    param = ConvertedFloatParameter(HyperParameter(float_parameter_dict))

    with monkeypatch.context() as m:
        m.setattr(param, "convert_log", True)

        with pytest.raises(ValueError):
            _convert_float(param, -1)

        value = _convert_float(param, 1)
        assert value == 0.0

    with monkeypatch.context() as m:
        m.setattr(param, "convert_log", False)
        value = _convert_float(param, 1)
        assert value == 1.0


def test_convert_float() -> None:
    param = ConvertedFloatParameter(HyperParameter(float_parameter_dict))

    value_by_convert_float = _convert_float(param, 1.0)
    value_by_convert_numerics = _convert_numerics(param, 1.0)

    assert value_by_convert_float == value_by_convert_numerics


def test_convert_int(monkeypatch: pytest.MonkeyPatch) -> None:
    param = ConvertedIntParameter(HyperParameter(int_parameter_dict))

    with monkeypatch.context() as m:
        m.setattr(param, "convert_int", True)
        value = _convert_int(param, 1)
        assert isinstance(value, float)
        assert value == 1.0

    with monkeypatch.context() as m:
        m.setattr(param, "convert_int", False)
        value = _convert_int(param, 1)
        assert isinstance(value, int)
        assert value == 1


def test_restore_float(monkeypatch: pytest.MonkeyPatch) -> None:
    param = ConvertedFloatParameter(HyperParameter(float_parameter_dict))

    with monkeypatch.context() as m:
        m.setattr(param, "convert_log", True)
        value = _restore_float(param, 0.0)
        assert value == 1.0

    with monkeypatch.context() as m:
        m.setattr(param, "convert_log", False)
        value = _restore_float(param, 1.0)
        assert value == 1.0


def test_restore_int(monkeypatch: pytest.MonkeyPatch) -> None:
    param = ConvertedIntParameter(HyperParameter(int_parameter_dict))

    with monkeypatch.context() as m:
        m.setattr(param, "convert_log", True)
        value = _restore_int(param, 0.0)
        assert isinstance(value, int)
        assert value == 1

    with monkeypatch.context() as m:
        m.setattr(param, "convert_log", False)
        value = _restore_int(param, 1)
        assert isinstance(value, int)
        assert value == 1


def test_is_weight_collected() -> None:
    param = WeightOfChoice(HyperParameter(categorical_parameter_dict), 0)
    weights = {
        param.original_name: {f"{param.name}_{i}": i * 0.1 for i in range(len(param.choices))}
    }
    assert _is_weight_collected(param, weights) is True
    weights = {
        param.original_name: {f"{param.name}_{i}": i * 0.1 for i in range(len(param.choices) - 1)}
    }
    assert _is_weight_collected(param, weights) is False


def test_make_weight_distribution() -> None:
    param = WeightOfChoice(HyperParameter(categorical_parameter_dict), 0)
    weights = {
        param.original_name: {f"{param.original_name}_{i}": i * 0.1 for i in range(len(param.choices))}
    }
    weight_distribution = _make_weight_distribution(param, weights)
    assert weight_distribution == [i * 0.1 for i in range(len(param.choices))]


def test_decode_weight_distribution() -> None:
    param = WeightOfChoice(HyperParameter(categorical_parameter_dict), 0)
    weight_distribution = [1] + [0] * (len(param.choices) - 1)
    choosed_value = _decode_weight_distribution(param, weight_distribution)
    assert choosed_value == param.choices[0]


def test_make_structured_value() -> None:
    param = ConvertedFloatParameter(HyperParameter(float_parameter_dict))
    value = param.lower
    structured_value = _make_structured_value(param, value)
    assert structured_value.get("parameter_name") == param.name
    assert structured_value.get("type") == param.type
    assert structured_value.get("value") == value
