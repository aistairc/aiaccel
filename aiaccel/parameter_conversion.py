from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, overload

import numpy as np

from aiaccel.parameter import HyperParameter, HyperParameterConfiguration


class ConvertedHyperparameter:
    name: str
    type: str
    lower: float
    upper: float
    choices: list[Any]
    sequence: list[Any]
    convert_log: bool
    convert_int: bool
    convert_choices: bool
    convert_sequence: bool
    choice_index: int

    def __init__(
        self,
        hyperparameter: HyperParameter,
        convert_log: bool = True,
        convert_int: bool = True,
        convert_choices: bool = True,
        convert_sequence: bool = True,
        choice_index: int = 0,
    ) -> None:
        """Conditions of hyperparameter of which the scale of numerical values,
        choices, or sequence are converted appropriately.

        Args:
        hyperparameters (list[HyperParameter]): A HyperPrameter object.
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

        Raises:
            TypeError: Causes when the parameter type is invalid.
        """
        self.name = hyperparameter.name
        # TODO: use data_type.
        self.type = (
            "uniform_float"
            if hyperparameter.type == "FLOAT"
            else "uniform_int"
            if hyperparameter.type == "INT"
            else "categorical"
            if hyperparameter.type == "CATEGORICAL"
            else "ordinal"
            if hyperparameter.type == "ORDINAL"
            else hyperparameter.type
        )
        self.convert_log = hyperparameter.log and convert_log
        self.convert_int = convert_int
        self.convert_choices = convert_choices
        self.convert_sequence = convert_sequence

        if self.type in ("uniform_int", "uniform_float"):
            self.lower = self.convert(hyperparameter.lower)
            self.upper = self.convert(hyperparameter.upper)
        elif self.type == "categorical":
            self.original_name = hyperparameter.name
            self.name = f"{hyperparameter.name}_{choice_index}"
            self.choices = hyperparameter.choices
            self.choice_index = choice_index
        elif self.type == "ordinal":
            self.original_name = hyperparameter.name
            self.name = f"{hyperparameter.name}_{choice_index}"
            self.sequence = hyperparameter.sequence
            self.choice_index = choice_index
        else:
            raise TypeError(f"Type of {self.name}: {self.type} is invalid.")

    def convert(self, external_value: Any) -> Any:
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
        if self.type == "uniform_int":
            if self.convert_log:
                if external_value <= 0:
                    raise ValueError("Log scaled value can not be negative.")
                return float(np.log(external_value)) if self.convert_int else int(np.log(external_value))
            else:
                return float(external_value) if self.convert_int else int(external_value)
        if self.type == "uniform_float":
            if self.convert_log:
                if external_value <= 0:
                    raise ValueError("Log scaled value can not be negative.")
                return np.log(external_value)
            else:
                return float(external_value)
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

    def convert_to_original_repr(self, internal_value: Any) -> Any:
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
            return self.choices[np.argmax(internal_value)] if self.convert_choices else internal_value
        elif self.type == "ordinal":
            return self.sequence[np.argmax(internal_value)] if self.convert_sequence else internal_value
        else:
            raise TypeError(f"Type of {self.name}: {self.type} is invalid.")


class ConvertedParameter(ABC):
    @overload
    def __init__(self, param: HyperParameter, convert_log: bool = True) -> None:
        ...

    @overload
    def __init__(self, param: HyperParameter, convert_log: bool = True, convert_int: bool = True) -> None:
        ...

    @overload
    def __init__(self, param: HyperParameter, convert_choices: bool = True, choice_index: int = -1) -> None:
        ...

    @overload
    def __init__(self, param: HyperParameter, convert_sequence: bool = True, choice_index: int = -1) -> None:
        ...

    def __init__(
        self,
        param: HyperParameter,
        convert_log: bool = True,
        convert_int: bool = True,
        convert_choices: bool = True,
        convert_sequence: bool = True,
        choice_index: int = -1,
    ) -> None:
        self.name = param.name
        self.type = param.type
        self.convert_log = convert_log
        self.convert_int = convert_int
        self.convert_choices = convert_choices
        self.convert_sequence = convert_sequence
        self.choice_index = choice_index

    @overload
    @abstractmethod
    def convert(self, external_value: float) -> float:
        ...

    @overload
    @abstractmethod
    def convert(self, external_value: Any) -> Any:
        ...

    @overload
    @abstractmethod
    def convert_to_original_repr(self, internal_value: float) -> float:
        ...

    @overload
    @abstractmethod
    def convert_to_original_repr(self, internal_value: float) -> int:
        ...

    @overload
    @abstractmethod
    def convert_to_original_repr(self, internal_value: list[float]) -> Any:
        ...


class NumericalConvertedParameter(ConvertedParameter):
    def __init__(self, param: HyperParameter, convert_log: bool = True, convert_int: bool = True) -> None:
        super().__init__(param, convert_log, convert_int)

    def convert(self, external_value: float) -> float:
        if self.convert_log:
            if external_value <= 0:
                raise ValueError("Log scaled value can not be negative.")
            return float(np.log(external_value))
        else:
            return float(external_value)

    def convert_to_original_repr(self, internal_value: float) -> float:
        if self.convert_log:
            return float(np.exp(internal_value))
        else:
            return float(internal_value)


class FloatConvertedParameter(NumericalConvertedParameter):
    def __init__(self, param: HyperParameter, convert_log: bool = True) -> None:
        super().__init__(param, convert_log)


class IntConvertedParameter(NumericalConvertedParameter):
    def __init__(self, param: HyperParameter, convert_log: bool = True, convert_int: bool = True) -> None:
        super().__init__(param, convert_log, convert_int)

    def convert(self, external_value: float) -> float:
        return int(super().convert(external_value)) if self.convert_int else super().convert(external_value)

    def convert_to_original_repr(self, internal_value: float) -> int:
        return int(super().convert_to_original_repr(internal_value))


class CategoricalConvertedParameter(ConvertedParameter):
    def __init__(self, param: HyperParameter, convert_choices: bool = True, choice_index: int = -1) -> None:
        super().__init__(param, convert_choices, choice_index)
        if convert_choices:
            self.name = f"{param.name}_{choice_index}"
        else:
            self.name = param.name
        self.original_name = param.name
        self.type = param.type
        self.convert_choices = convert_choices
        self.choices = param.choices

    def convert(self, internal_param: Any) -> Any:
        ...


class OrdinalConvertedParameter(CategoricalConvertedParameter):
    def __init__(self, param: HyperParameter, convert_sequence: bool = True, choice_index: int = -1) -> None:
        super().__init__(param, convert_sequence, choice_index)


class ConvertedHyperparameterConfiguration(HyperParameterConfiguration):
    """Collection of ConvertedHyperparameter objects.

    Args:
        hyperparameters (list[HyperParameter]): A list of HyperPrameter objects.
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
        hyperparameters: list[HyperParameter],
        convert_log: bool = True,
        convert_int: bool = True,
        convert_choices: bool = True,
        convert_sequence: bool = True,
    ) -> None:
        self._converted_params: dict[str, ConvertedHyperparameter] = {}
        for param in hyperparameters:
            if param.type in ("unifoem_float", "uniform_int"):
                self._converted_params[param.name] = ConvertedHyperparameter(
                    param, convert_log=convert_log, convert_int=convert_int
                )
            elif param.type == "categorical":
                for i in range(len(param.choices)):
                    self._converted_params[param.name] = ConvertedHyperparameter(
                        param, convert_choices=convert_choices, choice_index=i
                    )
            elif param.type == "ordinal":
                for i in range(len(param.sequence)):
                    self._converted_params[param.name] = ConvertedHyperparameter(
                        param, convert_sequence=convert_sequence, choice_index=i
                    )
            else:
                raise TypeError(f"Invalid type: {param.type}")

    def get_hyperparameter(self, name: str) -> ConvertedHyperparameter:
        """Gets a ConvertedHyperparameter object by specifying parameter name.

        Args:
            name (str): Name of parameter.

        Raises:
            KeyError: Causes when no parameter matches the given parameter name.

        Returns:
            ConvertedHyperparameter: Specified parameter name.
        """
        if name in self._converted_params:
            return self._converted_params[name]
        else:
            raise KeyError(f"Invalid parameter name: {name}")

    def get_parameter_list(self) -> list[ConvertedHyperparameter]:
        """Gets a list of parameters.

        Returns:
            list[ConvertedHyperparameter]: A list of ConvertedHyperparameter
                objects.
        """
        return list(self._converted_params.values())

    def get_parameter_dict(self) -> dict[str, ConvertedHyperparameter]:
        """Gets a dict object of ConvertedHyperparmaeters.

        Returns:
            dict[str, ConvertedHyperparameter]: A dict object of
                ConvertedHyperparameter objects.
        """
        return self._converted_params

    def to_original_repr(self, internal_params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Converts parameters in the original representation.

        Args:
            internal_params (list[dict[str, Any]]): A list of dict objects
                containing each parameter information in the internal
                representation.

        Returns:
            list[dict[str, Any]]: A list of dict objects containing each
                parameter information in the original representation.
        """

        params_in_original_repr: list[dict[str, Any]] = []
        choice_weights: dict[str, dict[str, Any]] = {}
        for param in internal_params:
            converted_param = self.get_hyperparameter(param["parameter_name"])
            if converted_param.type in ("uniform_float", "uniform_int"):
                params_in_original_repr.append(converted_param.convert_to_original_repr(param["value"]))
            elif converted_param.type == "categorical":
                if converted_param.convert_choices:
                    if converted_param.original_name in choice_weights:
                        choice_weights[converted_param.original_name][converted_param.name] = param["value"]
                    else:
                        choice_weights[converted_param.original_name] = {converted_param.name: param["value"]}
                    if len(choice_weights[converted_param.original_name]) == len(converted_param.choices):
                        encoded_choices = []
                        for i in range(len(converted_param.choices)):
                            encoded_choices.append(float(choice_weights[f"{converted_param.original_name}_{i}"]))
                        params_in_original_repr.append(converted_param.convert_to_original_repr(encoded_choices))
                else:
                    params_in_original_repr.append(converted_param.convert_to_original_repr(param["value"]))
            else:
                assert converted_param.type == "ordinal"
                if converted_param.convert_sequence:
                    if converted_param.original_name in choice_weights:
                        choice_weights[converted_param.original_name][converted_param.name] = param["value"]
                    else:
                        choice_weights[converted_param.original_name] = {converted_param.name: param["value"]}
                    if len(choice_weights[converted_param.original_name]) == len(converted_param.sequence):
                        encoded_sequence = []
                        for i in range(len(converted_param.sequence)):
                            encoded_sequence.append(float(choice_weights[f"{converted_param.original_name}_{i}"]))
                        params_in_original_repr.append(converted_param.convert_to_original_repr(encoded_sequence))
                else:
                    params_in_original_repr.append(converted_param.convert_to_original_repr(param["value"]))

        return params_in_original_repr
