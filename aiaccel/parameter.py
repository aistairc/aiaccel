from __future__ import annotations

from typing import Any

from numpy.random import RandomState


def get_type(parameter: dict[str, Any]) -> str:
    """Get a type of a specified parameter.

    Args:
        parameter (dict): A parameter dictionary in a configuration file.

    Returns:
        str: A parameter type any of 'INT', 'FLOAT', 'CATEGORICAL' and
        'ORDINAL'.
    """
    if parameter["type"].lower() == "uniform_int":
        return "INT"
    elif parameter["type"].lower() == "uniform_float":
        return "FLOAT"
    elif parameter["type"].lower() == "categorical":
        return "CATEGORICAL"
    elif parameter["type"].lower() == "ordinal":
        return "ORDINAL"
    else:
        return parameter["type"]


class HyperParameter(object):
    """
    A hyper parameter class.

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
        self.type = get_type(parameter)
        self.log = parameter.get("log", False)
        self.lower = parameter.get("lower", None)
        self.upper = parameter.get("upper", None)
        self.choices = parameter.get("choices", None)
        self.sequence = parameter.get("sequence", None)
        self.initial = parameter.get("initial", None)
        self.step = parameter.get("step", None)
        self.base = parameter.get("base", None)
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
        if initial and self.initial is not None:
            value = self.initial
        elif self.type.lower() == "int":
            value = rng.randint(self.lower, self.upper)
        elif self.type.lower() == "float":
            value = rng.uniform(self.lower, self.upper)
        elif self.type.lower() == "categorical":
            value = rng.choice(self.choices)
        elif self.type.lower() == "ordinal":
            value = rng.choice(self.sequence)
        else:
            raise TypeError(f"Invalid hyper parameter type: {self.type}")

        return {"name": self.name, "type": self.type, "value": value}


class HyperParameterConfiguration(object):
    """A configuration of hyper parameters.

    Args:
        json_string (dict): A configuration dictionary of hyper parameters.

    Attributes:
        json_string (dict): A configuration dictionary of hyper parameters.
        hps (dict): Hyper parameters.
    """

    def __init__(self, json_string: Any) -> None:
        self.json_string = json_string
        self.hps: dict[str, HyperParameter] = {}

        for hps in self.json_string:
            self.hps[hps["name"]] = HyperParameter(hps)

    def get_hyperparameter(self, name: str) -> HyperParameter:
        """Get a hyper parameter with a name.

        Args:
            name (str): A hyper parameter name.

        Returns:
            HyperParameter: A matched hyper parameter object.

        Raises:
            KeyError: Causes when no matched hyper parameter is.
        """
        if name in self.hps:
            return self.hps[name]
        else:
            raise KeyError(f"The parameter name {name} does not exist.")

    def get_parameter_list(self) -> list[HyperParameter]:
        """Get a list of hyper parameter objects.

        Returns:
            list[HyperParameter]: A list of hyper parameter objects.
        """
        return list(self.hps.values())

    def get_parameter_dict(self) -> dict[str, Any]:
        """Get a dictionary of hyper parameters.

        Returns:
            dict: A hyper parameter dictionary.
        """
        return self.hps

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
        for hyperparameter in self.hps.values():
            sampled_values.append(hyperparameter.sample(rng, initial))
        return sampled_values


def load_parameter(json_string: dict[str, Any]) -> HyperParameterConfiguration:
    """Load HyperParameterConfiguration object from a configuration file.

    Args:
        json_string (dict): A hyper parameter configuration.

    Returns:
        HyperParameterConfiguration: A hyper parameter configuration.
    """
    return HyperParameterConfiguration(json_string)
