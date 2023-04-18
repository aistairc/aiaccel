from __future__ import annotations

from typing import Any

from numpy.random import RandomState
from omegaconf.base import UnionNode
from omegaconf.nodes import (BooleanNode, BytesNode, FloatNode, IntegerNode,
                             PathNode, StringNode)


def is_uniform_float(data_type: str) -> bool:
    return data_type.lower() == 'uniform_float'


def is_uniform_int(data_type: str) -> bool:
    return data_type.lower() == 'uniform_int'


def is_categorical(data_type: str) -> bool:
    return data_type.lower() == 'categorical'


def is_ordinal(data_type: str) -> bool:
    return data_type.lower() == 'ordinal'


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
        self.name = parameter['name']
        self.type = parameter['type'].lower()
        self.log = parameter.get('log', False)
        self.lower = parameter.get('lower', None)
        self.upper = parameter.get('upper', None)
        self.choices = parameter.get('choices', None)
        if self.choices is not None:
            self.choices = [self.unwrap(v) for v in self.choices]
        self.sequence = parameter.get('sequence', None)
        if self.sequence is not None:
            self.sequence = [self.unwrap(v) for v in self.sequence]
        self.initial = parameter.get('initial', None)
        self.step = parameter.get('step', None)
        self.base = parameter.get('base', None)
        self.num_numeric_choices = parameter.get('num_numeric_choices', None)

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
                assert False, f'Invalid type: {type(value)}'
        elif isinstance(value, (IntegerNode, PathNode, StringNode, BooleanNode, BytesNode, FloatNode)):
            return value._value()
        else:
            return value


class IntParameter(AbstractParameter):
    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.randint(self.lower, self.upper)
        return {'name': self.name, 'type': self.type, 'value': self.unwrap(value)}


class FloatParameter(AbstractParameter):
    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.uniform(self.lower, self.upper)
        return {'name': self.name, 'type': self.type, 'value': self.unwrap(value)}


class CategoricalParameter(AbstractParameter):
    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.choice(self.choices)
        return {'name': self.name, 'type': self.type, 'value': self.unwrap(value)}


class OrdinalParameter(AbstractParameter):
    def sample(self, rng: RandomState, initial: bool = False) -> dict[str, Any]:
        if initial and self.initial is not None:
            value = self.initial
        else:
            value = rng.choice(self.sequence)
        return {'name': self.name, 'type': self.type, 'value': self.unwrap(value)}


class Parameter(AbstractParameter):
    def __new__(cls, parameter: dict[str, Any]) -> Any:
        data_type = parameter['type'].lower()
        if is_uniform_int(data_type):
            return IntParameter(parameter)
        elif is_uniform_float(data_type):
            return FloatParameter(parameter)
        elif is_categorical(data_type):
            return CategoricalParameter(parameter)
        elif is_ordinal(data_type):
            return OrdinalParameter(parameter)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
