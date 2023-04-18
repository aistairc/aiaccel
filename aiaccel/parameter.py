from __future__ import annotations

from typing import Any

from numpy.random import RandomState
from omegaconf.listconfig import ListConfig

from aiaccel.util.data_type import Parameter


class HyperParameterConfiguration(object):
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

    def get_parameter_dict(self) -> dict[str, Any]:
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
