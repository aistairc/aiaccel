from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest

from aiaccel.converted_parameter import ConvertedParameter, ConvertedParameterConfiguration
from aiaccel.parameter import HyperParameter

float_parameter_dict = {
    "name": "x0",
    "type": "uniform_float",
    "lower": 1.0,
    "upper": 2.0,
    "log": False,
}

int_parameter_dict = {
    "name": "x1",
    "type": "uniform_int",
    "lower": 1,
    "upper": 2,
    "log": False,
}

categorical_parameter_dict = {
    "name": "x2",
    "type": "categorical",
    "choices": ["a", "b", "c"]
}

ordinal_parameter_dict = {
    "name": "x3",
    "type": "ordinal",
    "sequence": [1, 2, 3]
}


class BaseTestParameterConversion:
    @pytest.fixture(autouse=True)
    def setup(self) -> Generator[None, None, None]:
        self.float_param = HyperParameter(float_parameter_dict)
        self.int_param = HyperParameter(int_parameter_dict)
        self.categorical_param = HyperParameter(categorical_parameter_dict)
        self.ordinal_param = HyperParameter(ordinal_parameter_dict)
        self.converted_float = ConvertedParameter(self.float_param)
        self.converted_int = ConvertedParameter(self.int_param)
        self.converted_categorical = ConvertedParameter(self.categorical_param)
        self.converted_ordinal = ConvertedParameter(self.ordinal_param)

        self.list_of_hyperparameters = [
            self.float_param,
            self.int_param,
            self.categorical_param,
            self.ordinal_param
        ]

        yield

        self.float_param = None
        self.int_param = None
        self.categorical_param = None
        self.ordinal_param = None
        self.converted_float = None
        self.converted_int = None
        self.converted_categorical = None
        self.converted_ordinal = None

        self.list_of_hyperparameters = None


class TestConvertedHyperparameter(BaseTestParameterConversion):
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        converted = ConvertedParameter(self.float_param)
        assert converted.type == "uniform_float"
        assert converted.convert_log is self.float_param.log
        assert converted.lower == self.float_param.lower
        assert converted.upper == self.float_param.upper

        with monkeypatch.context() as m:
            m.setattr(self.float_param, "log", True)

            converted = ConvertedParameter(self.float_param)
            assert converted.convert_log is True
            assert converted.lower == np.log(self.float_param.lower)
            assert converted.upper == np.log(self.float_param.upper)

            converted = ConvertedParameter(self.float_param, convert_log=False)
            assert converted.convert_log is False
            assert converted.lower == self.float_param.lower
            assert converted.upper == self.float_param.upper

        converted = ConvertedParameter(self.int_param)
        assert converted.type == "uniform_int"
        assert converted.convert_log is self.int_param.log
        assert converted.lower == self.int_param.lower
        assert converted.upper == self.int_param.upper

        with monkeypatch.context() as m:
            m.setattr(self.int_param, "log", True)

            converted = ConvertedParameter(self.int_param)
            assert converted.convert_log is True
            assert converted.lower == np.log(self.int_param.lower)
            assert converted.upper == np.log(self.int_param.upper)

            converted = ConvertedParameter(self.int_param, convert_log=False)
            assert converted.convert_log is False
            assert converted.lower == self.int_param.lower
            assert converted.upper == self.int_param.upper

        converted = ConvertedParameter(self.categorical_param)
        assert converted.type == "categorical"
        assert converted.convert_choices is True
        assert converted.choices == self.categorical_param.choices
        assert converted.lower == 0
        assert converted.upper == len(self.categorical_param.choices)

        converted = ConvertedParameter(self.categorical_param, convert_choices=False)
        assert converted.convert_choices is False
        assert converted.choices == self.categorical_param.choices
        assert hasattr(converted, "lower") is False
        assert hasattr(converted, "upper") is False

        converted = ConvertedParameter(self.ordinal_param)
        assert converted.type == "ordinal"
        assert converted.convert_sequence is True
        assert converted.sequence == self.ordinal_param.sequence
        assert converted.lower == 0
        assert converted.upper == len(self.ordinal_param.sequence)

        converted = ConvertedParameter(self.ordinal_param, convert_sequence=False)
        assert converted.convert_sequence is False
        assert converted.sequence == self.ordinal_param.sequence
        assert hasattr(converted, "lower") is False
        assert hasattr(converted, "upper") is False

        with monkeypatch.context() as m:
            m.setattr(self.float_param, "type", "invalid_type")
            with pytest.raises(TypeError):
                _ = ConvertedParameter(self.float_param)

    def test_convert_to_internal_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.converted_float.convert_to_internal_value(1) == 1

        with monkeypatch.context() as m:
            m.setattr(self.converted_float, "convert_log", True)
            assert self.converted_float.convert_to_internal_value(1) == 0
            with pytest.raises(ValueError):
                _ = self.converted_float.convert_to_internal_value(0)

        assert self.converted_int.convert_to_internal_value(1) == 1

        with monkeypatch.context() as m:
            m.setattr(self.converted_int, "convert_log", True)
            assert self.converted_int.convert_to_internal_value(1) == 0
            with pytest.raises(ValueError):
                _ = self.converted_int.convert_to_internal_value(0)

        assert self.converted_categorical.convert_to_internal_value(
            self.categorical_param.choices[0]) == 0

        with pytest.raises(ValueError):
            _ = self.converted_categorical.convert_to_internal_value("z")

        with monkeypatch.context() as m:
            m.setattr(self.converted_categorical, "convert_choices", False)
            assert self.converted_categorical.convert_to_internal_value(
                self.categorical_param.choices[0]) == self.categorical_param.choices[0]

        assert self.converted_ordinal.convert_to_internal_value(
            self.ordinal_param.sequence[0]) == 0

        with pytest.raises(ValueError):
            _ = self.converted_ordinal.convert_to_internal_value(9)

        with monkeypatch.context() as m:
            m.setattr(self.converted_ordinal, "convert_sequence", False)
            assert self.converted_ordinal.convert_to_internal_value(
                self.ordinal_param.sequence[0]) == self.ordinal_param.sequence[0]

        with monkeypatch.context() as m:
            m.setattr(self.converted_float, "type", "invalid_type")
            with pytest.raises(TypeError):
                _ = self.converted_float.convert_to_internal_value(1)

    def test_convert_to_external_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self.converted_float.convert_to_external_value(1) == float(1)

        with monkeypatch.context() as m:
            m.setattr(self.converted_float, "convert_log", True)
            assert self.converted_float.convert_to_external_value(1) == np.exp(1)

        assert self.converted_int.convert_to_external_value(1) == int(1)

        with monkeypatch.context() as m:
            m.setattr(self.converted_int, "convert_log", True)
            assert self.converted_int.convert_to_external_value(1) == int(np.exp(1))

        assert self.converted_categorical.convert_to_external_value(
            0.5) == self.categorical_param.choices[int(0.5)]

        with monkeypatch.context() as m:
            m.setattr(self.converted_categorical, "convert_choices", False)
            assert self.converted_categorical.convert_to_external_value(0.5) == 0.5

        assert self.converted_ordinal.convert_to_external_value(
            0.5) == self.ordinal_param.sequence[int(0.5)]

        with monkeypatch.context() as m:
            m.setattr(self.converted_ordinal, "convert_sequence", False)
            assert self.converted_ordinal.convert_to_external_value(0.5) == 0.5

        with monkeypatch.context() as m:
            m.setattr(self.converted_float, "type", "invalid_type")
            with pytest.raises(TypeError):
                _ = self.converted_float.convert_to_external_value(999)


class TestConvertedHyperparameterConfiguraion(BaseTestParameterConversion):
    def test_init(self) -> None:
        converted_parameters = ConvertedParameterConfiguration(
            self.list_of_hyperparameters, convert_log=True, convert_choices=True, convert_sequence=True
        )
        assert len(self.list_of_hyperparameters) == len(converted_parameters.converted_parameters)
        for converted_param in converted_parameters.converted_parameters.values():
            if converted_param.type == "uniform_float":
                assert converted_param.convert_log == self.float_param.log
            elif converted_param.type == "uniform_int":
                assert converted_param.convert_log == self.int_param.log
            elif converted_param.type == "categorical":
                assert converted_param.convert_choices is True
            else:
                assert converted_param.type == "ordinal"
                assert converted_param.convert_sequence is True

        converted_parameters = ConvertedParameterConfiguration(
            self.list_of_hyperparameters, convert_log=False, convert_choices=False, convert_sequence=False
        )
        assert len(self.list_of_hyperparameters) == len(converted_parameters.converted_parameters)
        for converted_param in converted_parameters.converted_parameters.values():
            if converted_param.type == "uniform_float":
                assert converted_param.convert_log is False
            elif converted_param.type == "uniform_int":
                assert converted_param.convert_log is False
            elif converted_param.type == "categorical":
                assert converted_param.convert_choices is False
            else:
                assert converted_param.type == "ordinal"
                assert converted_param.convert_sequence is False

    def test_get(self):
        converted_parameters = ConvertedParameterConfiguration(
            self.list_of_hyperparameters
        )
        assert converted_parameters.get("x0").name == "x0"

        with pytest.raises(KeyError):
            _ = converted_parameters.get("x9")
