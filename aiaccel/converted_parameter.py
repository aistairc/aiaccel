from __future__ import annotations

from collections.abc import Iterable
from logging import getLogger
from typing import Any

import numpy as np
from numpy.random import RandomState

from aiaccel.parameter import (
    AbstractParameter,
    CategoricalParameter,
    FloatParameter,
    HyperParameterConfiguration,
    IntParameter,
    OrdinalParameter,
    Parameter,
)


class ConvertedParameter(AbstractParameter):
    def __init__(self, param: Parameter) -> None:
        self.name = param.name
        self.type = param.type
        self.original_initial = param.initial
        self.initial = param.initial


class ConvertedNumericalParameter(ConvertedParameter):
    def __init__(self, param: Parameter, convert_log: bool = True):
        super().__init__(param)
        self.convert_log = convert_log and param.log
        self.original_lower = param.lower
        self.original_upper = param.upper


class ConvertedFloatParameter(ConvertedNumericalParameter):
    def __init__(self, param: Parameter, convert_log: bool = True):
        super().__init__(param, convert_log)
        self.lower = _convert_float(self, param.lower)
        self.upper = _convert_float(self, param.upper)
        if isinstance(param.initial, Iterable):  # For Nelder-Mead
            self.initial = [_convert_float(self, value) for value in param.initial]
        else:  # For others
            self.initial = _convert_float(self, param.initial) if param.initial is not None else None

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, str | float]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.uniform(self.lower, self.upper)
        return {"name": self.name, "type": self.type, "value": value}


class ConvertedIntParameter(ConvertedNumericalParameter):
    def __init__(self, param: Parameter, convert_log: bool = True, convert_int: bool = True) -> None:
        super().__init__(param, convert_log)
        self.convert_int = convert_int
        self.lower = _convert_int(self, param.lower)
        self.upper = _convert_int(self, param.upper)
        if isinstance(param.initial, Iterable):  # For Nelder-Mead
            self.initial = [_convert_int(self, value) for value in param.initial]
        else:  # For others
            self.initial = _convert_int(self, param.initial) if param.initial is not None else None

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, str | float]:
        if initial and self.initial is not None:
            value = self.initial
        elif self.convert_int:
            value = rng.uniform(self.lower, self.upper)
        else:
            assert self.convert_int is False
            value = rng.randint(self.lower, self.upper)
        return {"name": self.name, "type": self.type, "value": value}


class ConvertedCategoricalParameter(ConvertedParameter):
    def __init__(self, param: Parameter, convert_choices: bool = True) -> None:
        super().__init__(param)
        self.choices = param.choices
        self.convert_choices = convert_choices

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.choice(self.choices)
        return {"name": self.name, "type": self.type, "value": value}


class ConvertedOrdinalParameter(ConvertedParameter):
    def __init__(self, param: Parameter, convert_sequence: bool = True) -> None:
        super().__init__(param)
        self.sequence = param.sequence
        self.convert_sequence = convert_sequence

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.choice(self.sequence)
        return {"name": self.name, "type": self.type, "value": value}


class WeightOfChoice(ConvertedParameter):
    def __init__(self, param: Parameter, choice_index: int) -> None:
        super().__init__(param)
        self.choice_index = choice_index
        self.original_name = self.name
        self.name = _make_weight_name(self.original_name, choice_index)
        self.choices = list(param.choices if param.type.lower() == "categorical" else param.sequence)
        self.lower = 0.0
        self.upper = 1.0
        if isinstance(param.initial, Iterable) and not isinstance(param.initial, str):  # For Nelder-Mead
            self.initial = [float(list(self.choices).index(value) == self.choice_index) for value in param.initial]
        else:  # For others
            self.initial = (
                float(list(self.choices).index(self.original_initial) == self.choice_index)
                if self.original_initial is not None
                else None
            )

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, str | float]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.uniform(self.lower, self.upper)
        return {"name": self.name, "type": self.type, "value": value}


class ConvertedParameterConfiguration(HyperParameterConfiguration):
    """Collection of ConvertedParameter objects.

    Args:
        params (HyperParameterConfiguration): A HyperParameterConfiguration
            object.
        convert_log (bool, optional): Whether to convert the numerical values
            between log and linear scale when log of the hyperparameter object
            is True. Defaults to True.
        convert_int (bool, optional): Whether to convert the int value to
            float. For example, if convert_int = False and log-scale conversion
            is enabled, the value `v` will be converted as `int(numpy.log(v))`.
            Defaults to True.
        convert_choices (bool, optional): Whether to treat the choices of
            categorical parameter as float value corresponding to the index of
            choices. Defaults to True.
        convert_sequence (bool, optional): Whether to treat the sequence of
            ordinal parameter as a float value corresponding to the index of
            sequence. Defaults to True.
    """

    def __init__(
        self,
        params: HyperParameterConfiguration,
        convert_log: bool = True,
        convert_int: bool = True,
        convert_choices: bool = True,
        convert_sequence: bool = True,
    ) -> None:
        self._converted_params = self.convert(params, convert_log, convert_int, convert_choices, convert_sequence)

    def convert(
        self,
        params: HyperParameterConfiguration,
        convert_log: bool = True,
        convert_int: bool = True,
        convert_choices: bool = True,
        convert_sequence: bool = True,
    ) -> dict[str, AbstractParameter]:
        """Converts all of HyperParameter in the given list to the internal
        representation.

        If `categorical` (`ordinal`) parameters are included, and
        `convert_choice` (`convert_sequence`) is `True`, the number of
        parameters in a returned list may different from the original one.
        This is because, each choice in `choices` (`sequence`) is treated as a
        float parameter bounded by lower and upper value of 0.0 and 1.0,
        respectively.

        Args:
            params (list[HyperParameter]): A list of HyperPrameter objects.
            convert_log (bool, optional): Whether to convert the numerical
                values between log and linear scale when log of the
                hyperparameter object is True. Defaults to True.
            convert_int (bool, optional): Whether to convert the int value to
                float. For example, if convert_int = False and log-scale
                conversion is enabled, the value `v` will be converted as
                `int(numpy.log(v))`. Defaults to True.
            convert_choices (bool, optional): Whether to treat the choices of
                categorical parameter as float value corresponding to the index
                of choices. Defaults to True.
            convert_sequence (bool, optional): Whether to treat the sequence of
                ordinal parameter as a float value corresponding to the index
                of sequence. Defaults to True.

        Raises:
            TypeError: Causes when the type of parameter is invalid.

        Returns:
            dict[str, HyperParameter]: A dict object of ConvertedParameter.
                Keys of the dict specify internaly effective names of
                parameters.
        """
        converted_params: dict[str, AbstractParameter] = {}
        for param in params.get_parameter_list():
            if isinstance(param, FloatParameter):
                converted_params[param.name] = ConvertedFloatParameter(param, convert_log=convert_log)
            elif isinstance(param, IntParameter):
                converted_params[param.name] = ConvertedIntParameter(
                    param, convert_log=convert_log, convert_int=convert_int
                )
            elif isinstance(param, CategoricalParameter):
                if convert_choices:
                    for i in range(len(param.choices)):
                        converted_param = WeightOfChoice(param, choice_index=i)
                        converted_params[converted_param.name] = converted_param
                    logger = getLogger("root.optimizer")
                    logger.warning(
                        f"The choices of {param.name} ({param.type}) is converted to "
                        f"{len(param.choices)} float parameters."
                    )
                else:
                    converted_params[param.name] = ConvertedCategoricalParameter(param, convert_choices=convert_choices)
            elif isinstance(param, OrdinalParameter):
                if convert_sequence:
                    for i in range(len(param.sequence)):
                        converted_param = WeightOfChoice(param, choice_index=i)
                        converted_params[converted_param.name] = converted_param
                    logger = getLogger("root.optimizer")
                    logger.warning(
                        f"The sequence of {param.name} ({param.type}) is converted to "
                        f"{len(param.sequence)} float parameters."
                    )
                else:
                    converted_params[param.name] = ConvertedOrdinalParameter(param, convert_sequence=convert_sequence)
            else:
                raise TypeError(f"Invalid type: {param.type}")

        return converted_params

    def to_original_repr(self, params_in_internal_repr: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Converts parameters in the original representation.

        Args:
            params_in_internal_expr (list[dict[str, Any]]): A list of dict objects
                containing each parameter information in the internal
                representation.

        Returns:
            list[dict[str, Any]]: A list of dict objects containing each
                parameter information in the original representation.
        """

        params_in_original_repr: list[dict[str, Any]] = []
        weights: dict[str, dict[str, Any]] = {}
        for param in params_in_internal_repr:
            converted_param = self.get_hyperparameter(param["parameter_name"])
            if isinstance(converted_param, ConvertedFloatParameter):
                value = _restore_float(converted_param, param["value"])
                params_in_original_repr.append(_make_structured_value(converted_param, value))
            elif isinstance(converted_param, ConvertedIntParameter):
                value = _restore_int(converted_param, param["value"])
                params_in_original_repr.append(_make_structured_value(converted_param, value))
            elif isinstance(converted_param, ConvertedCategoricalParameter):
                value = param["value"]
                params_in_original_repr.append(_make_structured_value(converted_param, value))
            elif isinstance(converted_param, ConvertedOrdinalParameter):
                value = param["value"]
                params_in_original_repr.append(_make_structured_value(converted_param, value))
            elif isinstance(converted_param, WeightOfChoice):
                if converted_param.original_name in weights:
                    weights[converted_param.original_name][converted_param.name] = param["value"]
                else:
                    weights[converted_param.original_name] = {converted_param.name: param["value"]}
                if _is_weight_collected(converted_param, weights):
                    weight_distribution = _make_weight_distribution(converted_param, weights)
                    value = _decode_weight_distribution(converted_param, weight_distribution)
                    params_in_original_repr.append(_make_structured_value(converted_param, value))
            else:
                raise TypeError(f"Invalid type: {type(converted_param).__name__}")

        return params_in_original_repr

    def sample(self, rng: RandomState, initial: bool = False) -> list[dict[str, Any]]:
        sampled_values = [param.sample(rng, initial) for param in self._converted_params.values()]
        return sampled_values

    def get_hyperparameter(self, name: str) -> Any:  # TODO fix type
        """Gets a ConvertedParameter object by specifying parameter name.

        Args:
            name (str): Name of parameter.

        Raises:
            KeyError: Causes when no parameter matches the given parameter name.

        Returns:
            ConvertedParameter: Specified parameter name.
        """
        if name in self._converted_params:
            return self._converted_params[name]
        else:
            raise KeyError(f"Invalid parameter name: {name}")

    def get_parameter_list(self) -> list[Any]:  # TODO fix type
        """Gets a list of parameters.

        Returns:
            list[ConvertedParameter]: A list of ConvertedParameter
                objects.
        """
        return list(self._converted_params.values())

    def get_parameter_dict(self) -> dict[str, Any]:  # TODO fix type
        """Gets a dict object of ConvertedHyperparmaeters.

        Returns:
            dict[str, ConvertedParameter]: A dict object of
                ConvertedParameter objects.
        """
        return self._converted_params

    def get_parameter_names(self) -> list[str]:
        return list(self.get_parameter_dict().keys())

    def get_empty_parameter_dict(self) -> list[dict[str, Any]]:
        base_params = []
        for param in self.get_parameter_list():
            base_params.append({"parameter_name": param.name, "type": param.type, "value": None})
        return base_params


def _make_weight_name(original_name: str, choice_index: int) -> str:
    """Makes name of internal parameter for weight of one of the choices.

    Args:
        original_name (str): The name of categorical or ordinal parameter.
        choice_index (str): The index of choice.

    Returns:
        str: Name of internal parameter for weight of one of the choices.
    """
    return f"{original_name}_{choice_index}"


def _convert_numerics(param: ConvertedFloatParameter | ConvertedIntParameter, external_value: float) -> float:
    if param.convert_log:
        if external_value <= 0:
            raise ValueError("Log scaled value can not be negative.")
        return float(np.log(external_value))
    else:
        return external_value


def _convert_float(param: ConvertedFloatParameter, external_value: float) -> float:
    return _convert_numerics(param, external_value)


def _convert_int(param: ConvertedIntParameter, external_value: int) -> float:
    converted_value = _convert_numerics(param, external_value)
    if param.convert_int:
        return float(converted_value)
    else:
        return int(converted_value)


def _restore_float(param: ConvertedFloatParameter, internal_value: float) -> float:
    if param.convert_log:
        return float(np.exp(internal_value))
    else:
        return internal_value


def _restore_int(param: ConvertedIntParameter, internal_value: float | int) -> int:
    if param.convert_log:
        return int(np.exp(internal_value))
    else:
        return int(internal_value)


def _is_weight_collected(param: WeightOfChoice, weights: dict[str, dict[str, str | float]]) -> bool:
    return len(param.choices) == len(weights[param.original_name])


def _make_weight_distribution(param: WeightOfChoice, weights: dict[str, dict[str, str | float]]) -> list[float]:
    original_name = param.original_name
    weight_dict = weights[original_name]
    weight_list = []
    for i in range(len(param.choices)):
        weight_list.append(float(weight_dict[f"{original_name}_{i}"]))
    return weight_list


def _decode_weight_distribution(param: WeightOfChoice, weight_distribution: list[float]) -> Any:
    return param.choices[np.argmax(weight_distribution)]


def _make_structured_value(param: ConvertedParameter, value: Any) -> dict[str, Any]:
    return {
        "parameter_name": param.original_name if isinstance(param, WeightOfChoice) else param.name,
        "type": param.type,
        "value": value,
    }
