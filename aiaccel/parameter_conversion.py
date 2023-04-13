from __future__ import annotations

from typing import Any, Literal

import numpy as np

from aiaccel.parameter import HyperParameter


class ConvertedHyperparameter:
    name: str
    type: Literal["uniform_int", "uniform_float", "categorical", "ordinal"]
    lower: float
    upper: float
    choices: list[Any]
    sequence: list[Any]
    convert_log: bool
    convert_choices: bool
    convert_sequence: bool

    def __init__(self, hyperparameter: HyperParameter,
                 convert_log: bool = True, convert_choices: bool = True,
                 convert_sequence: bool = True) -> None:
        """Conditions of hyperparameter of which the scale of numerical values,
        choices, or sequence are converted appropriately.

        Args:
        hyperparameters (list[HyperParameter]): A HyperPrameter object.
        convert_log (bool, optional): Whether to convert the numerical values
            between log and linear scale when log of the hyperparameter object
            is True. Defaults to True.
        convert_choices (bool, optional): Whether to treat the choices of
            categorical parameter as float value corresponding to the index of
            choices. Defaults to True.
        convert_sequence (bool, optional): Whether to treat the sequence of
            ordinal parameter as a float value corresponding to the index of
            sequence. Defaults to True.
        """
        self.name = hyperparameter.name
        self.type = hyperparameter.type
        self.convert_log = hyperparameter.log and convert_log
        self.convert_choices = convert_choices
        self.convert_sequence = convert_sequence

        if self.type in ("uniform_int", "uniform_float"):
            self.lower = self.convert_to_internal_value(hyperparameter.lower)
            self.upper = self.convert_to_internal_value(hyperparameter.upper)
        elif self.type == "categorical":
            self.choices = hyperparameter.choices
            if self.convert_choices:
                self.lower = 0
                self.upper = len(self.choices)
        elif self.type == "ordinal":
            self.sequence = hyperparameter.sequence
            if self.convert_sequence:
                self.lower = 0
                self.upper = len(self.sequence)

    def convert_to_internal_value(self, external_value: Any) -> Any:
        """Converts a value in the external representation to the internal
        representation.

        Args:
            external_value (Any): A value which should be in the external
                representation.

        Raises:
            ValueError: Causes when external_value is invalid.
            TypeError: Causes when the parameter type is invalid.

        Returns:
            Any: A value in the internal representation.
        """
        if self.type in ("uniform_int", "uniform_float"):
            if external_value <= 0:
                raise ValueError("Log scaled value can not be negative.")
            return np.log(external_value) if self.convert_log else external_value
        elif self.type == "categorical":
            if external_value not in self.choices:
                raise ValueError(f"Specified value: {external_value} is not in choices.")
            return self.choices.index(external_value) if self.convert_choices else external_value
        elif self.type == "ordinal":
            if external_value not in self.sequence:
                raise ValueError(f"Specified value: {external_value} is not in sequence.")
            return self.sequence.index(external_value) if self.convert_sequence else external_value
        else:
            raise TypeError(f"Type of {self.name}: {self.type} is invalid.")

    def convert_to_external_value(self, internal_value: Any) -> Any:
        """Converts a value in the internal representation to the external
        representation.

        Args:
            internal_value (Any): A value which should be in the internal
                representation.

        Raises:
            TypeError: Causes when the parameter type is invalid.

        Returns:
            Any: A value in the external representation.
        """

        if self.type == "uniform_int":
            return int(np.exp(internal_value) if self.convert_log else internal_value)
        elif self.type == "uniform_float":
            return float(np.exp(internal_value) if self.convert_log else internal_value)
        elif self.type == "categorical":
            return self.choices[int(internal_value)] if self.convert_choices else internal_value
        elif self.type == "ordinal":
            return self.sequence[int(internal_value)] if self.convert_sequence else internal_value
        else:
            raise TypeError(f"Type of {self.name}: {self.type} is invalid.")


class ConvertedHyperparameterConfiguration:
    """Collection of ConvertedHyperparameter objects.

    Args:
        hyperparameters (list[HyperParameter]): A list of HyperPrameter objects.
        convert_log (bool, optional): Whether to convert the numerical values
            between log and linear scale when log of the hyperparameter object
            is True. Defaults to True.
        convert_choices (bool, optional): Whether to treat the choices of
            categorical parameter as float value corresponding to the index of
            choices. Defaults to True.
        convert_sequence (bool, optional): Whether to treat the sequence of
            ordinal parameter as a float value corresponding to the index of
            sequence. Defaults to True.

    Attributes:
        converted_parameters (dict[str, ConvertedHyperparameter]): A dict
            object of ConvertedHyperparameter.
    """

    def __init__(self, hyperparameters: list[HyperParameter],
                 convert_log: bool = True, convert_choices: bool = True,
                 convert_sequence: bool = True) -> None:
        self.converted_parameters: dict[str, ConvertedHyperparameter] = {}
        for param in hyperparameters:
            self.converted_parameters[param.name] = ConvertedHyperparameter(
                param, convert_log, convert_choices, convert_sequence)

    def get(self, name: str) -> ConvertedHyperparameter:
        """Gets a ConvertedHyperparameter object by specifying parameter name.

        Args:
            name (str): Name of parameter.

        Raises:
            KeyError: Causes when no parameter matches the given parameter name.

        Returns:
            ConvertedHyperparameter: Specified parameter name.
        """
        if name in self.converted_parameters:
            return self.converted_parameters[name]
        else:
            raise KeyError(f"Invalid parameter name: {name}")
