from __future__ import annotations

from typing import Any
from collections.abc import Iterable

from numpy.random import RandomState
from omegaconf.base import UnionNode
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import BooleanNode, BytesNode, FloatNode, IntegerNode, PathNode, StringNode


def is_uniform_float(data_type: str) -> bool:
    return data_type.lower() == "uniform_float"


def is_uniform_int(data_type: str) -> bool:
    return data_type.lower() == "uniform_int"


def is_categorical(data_type: str) -> bool:
    return data_type.lower() == "categorical"


def is_ordinal(data_type: str) -> bool:
    return data_type.lower() == "ordinal"


def is_within_range(initial_value: int | float, lower: int | float, upper: int | float):
    return lower <= initial_value and initial_value <= upper


def is_in_category(initial_value: Any, category_list: list):
    return initial_value in category_list


class AbstractParameter:
    """
    A parameter class.

    Args:
        parameter (dict): A parameter dictionary in a configuration file.

    Attributes:
        _raw_dict (dict): A parameter dictionary in a configuration file.
        name (str): A parameter name.
        type (str): A parameter type any of 'uniform_int', 'uniform_float',
            'categorical' and 'ordinal'.
        log (bool): A parameter is logarithm or not.
        lower (float | int): A lower value of a parameter.
        upper (float | int): A upper value of a parameter.
        choices (list[float, int, str]): This is set as a list of a parameter,
            when a parameter type is 'categorical'.
        sequence (list[float, int, str]): This is set as a list of a parameter,
            when a parameter type is 'ordinal'.
        initial (float | int | str): A initial value. If this is set, this
            value is evaluated at first run.
    """

    def __init__(self, parameter: dict[str, Any]) -> None:
        self._raw_dict = parameter
        self.name = parameter["name"]
        self.type = parameter["type"].lower()
        self.log = parameter.get("log", False)
        self.lower = parameter.get("lower", None)
        self.upper = parameter.get("upper", None)
        self.initial = parameter.get("initial", None)
        self.step = parameter.get("step", None)
        self.base = parameter.get("base", None)
        self.choices = parameter.get("choices", None)
        self.sequence = parameter.get("sequence", None)
        self.num_numeric_choices = parameter.get("num_numeric_choices", None)

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        """Sample a parameter.

        Args:
            rng (RandomState): A random generator.
            initial (bool, optional): Whether to require a initial value. If
                True, returns the initial value. Defaults to False.

        Returns:
            dict: A parameter dictionary.

        Raises:
            TypeError: Causes when an invalid type is set.
        """
        raise NotImplementedError

    def unwrap(self, value: Any) -> Any:
        if isinstance(value, UnionNode):
            value = value._value()
            if isinstance(value, (IntegerNode, PathNode, StringNode, BooleanNode, BytesNode, FloatNode)):
                return value._value()
            else:
                assert False, f"Invalid type: {type(value)}"
        elif isinstance(value, (IntegerNode, PathNode, StringNode, BooleanNode, BytesNode, FloatNode)):
            return value._value()
        else:
            return value


class IntParameter(AbstractParameter):
    def __init__(self, parameter: dict[str, Any]) -> None:
        super().__init__(parameter)
        if isinstance(self.initial, Iterable):  # For Nelder-Mead
            for value in self.initial:
                if not is_within_range(value, self.lower, self.upper):
                    assert False, "initial is out of range"
        elif self.initial is not None and not is_within_range(self.initial, self.lower, self.upper):
            assert False, "initial is out of range"

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.randint(self.lower, self.upper)
        return {"name": self.name, "type": self.type, "value": self.unwrap(value)}


class FloatParameter(AbstractParameter):
    def __init__(self, parameter: dict[str, Any]) -> None:
        super().__init__(parameter)
        if isinstance(self.initial, Iterable):  # For Nelder-Mead
            for value in self.initial:
                if not is_within_range(value, self.lower, self.upper):
                    assert False, "initial is out of range"
        elif self.initial is not None and not is_within_range(self.initial, self.lower, self.upper):
            assert False, "initial is out of range"

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.uniform(self.lower, self.upper)
        return {"name": self.name, "type": self.type, "value": self.unwrap(value)}


class CategoricalParameter(AbstractParameter):
    def __init__(self, parameter: dict[str, Any]) -> None:
        super().__init__(parameter)
        if self.choices is not None:
            self.choices = [self.unwrap(v) for v in self.choices]

        if self.initial is not None and not is_in_category(self.initial, self.choices):
            assert False, "initial is not included in choices"

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.choice(self.choices)
        return {"name": self.name, "type": self.type, "value": self.unwrap(value)}


class OrdinalParameter(AbstractParameter):
    def __init__(self, parameter: dict[str, Any]) -> None:
        super().__init__(parameter)
        if self.sequence is not None:
            self.sequence = [self.unwrap(v) for v in self.sequence]

        if self.initial is not None and not is_in_category(self.initial, self.sequence):
            assert False, "initial is not included in sequence"

    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.choice(self.sequence)
        return {"name": self.name, "type": self.type, "value": self.unwrap(value)}


class Parameter(AbstractParameter):
    def __new__(cls, parameter: dict[str, Any]) -> Any:
        data_type = parameter["type"].lower()
        if is_uniform_int(data_type):
            return IntParameter(parameter)
        elif is_uniform_float(data_type):
            return FloatParameter(parameter)
        elif is_categorical(data_type):
            return CategoricalParameter(parameter)
        elif is_ordinal(data_type):
            return OrdinalParameter(parameter)
        else:
            raise TypeError(f"Invalid data_type: {data_type}")


class HyperParameterConfiguration:
    """A configuration of hyper parameters.

    Args:
        parameters (ListConfig): A configuration dictionary of hyperparameters.

    Attributes:
        param (dict): A dictionary containing hyperparameter configurations.
    """

    def __init__(self, parameters: ListConfig) -> None:
        self.param: dict[str, Parameter] = {}

        for param in parameters:
            self.param[param["name"]] = Parameter(param)

    def get_hyperparameter(self, name: str) -> Parameter:
        """Get a hyper parameter with a name.

        Args:
            name (str): A hyper parameter name.

        Returns:
            HyperParameter: A matched hyper parameter object.

        Raises:
            KeyError: Causes when no matched hyper parameter is.
        """
        if name in self.param:
            return self.param[name]
        else:
            raise KeyError(f"The parameter name {name} does not exist.")

    def get_parameter_list(self) -> list[Parameter]:
        """Get a list of hyper parameter objects.

        Returns:
            list[HyperParameter]: A list of hyper parameter objects.
        """
        return list(self.param.values())

    def get_parameter_dict(self) -> dict[str, Parameter]:
        """Get a dictionary of hyper parameters.

        Returns:
            dict: A hyper parameter dictionary.
        """
        return self.param

    def sample(self, rng: RandomState, initial: bool = False) -> list[dict[str, Any]]:
        """Sample a hyper parameters set.

        Args:
            rng (RandomState): A random generator.
            initial (bool, optional): Whether to require a initial value. If
                True, returns the initial value. Defaults to False.

        Returns:
            list[dict]: A hyper parameters set.
        """
        sampled_values = []
        for parameter in self.param.values():
            sampled_values.append(parameter.sample(rng, initial))
        return sampled_values


def load_parameter(parameters: ListConfig) -> HyperParameterConfiguration:
    """Load HyperParameterConfiguration object from a configuration file.

    Args:
        parameters (dict): A hyper parameter object.

    Returns:
        HyperParameterConfiguration: A hyper parameter configuration.
    """
    return HyperParameterConfiguration(parameters)
