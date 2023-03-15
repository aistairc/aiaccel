from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from numpy.random import RandomState

import aiaccel
from aiaccel.util.filesystem import load_yaml


def get_best_parameter(files: list[Path], goal: str, dict_lock: Path
                       ) -> tuple[float | None, Path | None]:
    """Get a best parameter in specified files.

    Args:
        files (list[Path]): A list of files to find a best.
        goal (str): Maximize or Minimize.
        dict_lock (Path): A directory to store lock files.

    Returns:
        tuple[float | None, Path | None]: A best result value and a
        file path. It returns None if a number of files is less than one.

    Raises:
        ValueError: Causes when an invalid goal is set.
    """

    if len(files) < 1:
        return None, None

    yml = load_yaml(files[0], dict_lock)

    try:
        best = float(yml['result'])
    except TypeError:
        logger = logging.getLogger('root.master.parameter')
        logger.error(f'Invalid result: {yml["result"]}.')
        return None, None

    best_file = files[0]

    for f in files[1:]:
        yml = load_yaml(f, dict_lock)
        result = float(yml['result'])

        if goal.lower() == aiaccel.goal_maximize:
            if best < result:
                best, best_file = result, f
        elif goal.lower() == aiaccel.goal_minimize:
            if best > result:
                best, best_file = result, f
        else:
            logger = logging.getLogger('root.master.parameter')
            logger.error(f'Invalid goal: {goal}.')
            raise ValueError(f'Invalid goal: {goal}.')

    return best, best_file


def get_type(parameter: dict[str, Any]) -> str:
    """Get a type of a specified parameter.

    Args:
        parameter (dict): A parameter dictionary in a configuration file.

    Returns:
        str: A parameter type any of 'INT', 'FLOAT', 'CATEGORICAL' and
        'ORDINAL'.
    """
    if parameter['type'].lower() == 'uniform_int':
        return 'INT'
    elif parameter['type'].lower() == 'uniform_float':
        return 'FLOAT'
    elif parameter['type'].lower() == 'categorical':
        return 'CATEGORICAL'
    elif parameter['type'].lower() == 'ordinal':
        return 'ORDINAL'
    else:
        return parameter['type']


class HyperParameter(object):
    """
    A hyper parameter class.

    Args:
        parameter (dict): A parameter dictionary in a configuration file.

    Attributes:
        _raw_dict (dict): A parameter dictionary in a configuration file.
        name (str): A parameter name.
        type (str): A parameter type any of 'INT', 'FLOAT', 'CATEGORICAL' and
            'ORDINAL'.
        log (bool): A parameter is logarithm or not.
        lower (float | int): A lower value of a parameter.
        upper (float | int): A upper value of a parameter.
        choices (list[float, int, str]): This is set as a list of a parameter,
            when a parameter type is 'CATEGORICAL'.
        sequence (list[float, int, str]): This is set as a list of a parameter,
            when a parameter type is 'ORDINAL'.
        initial (float | int | str): A initial value. If this is set, this
            value is evaluated at first run.
        q (float | int): A quantization factor.
    """

    def __init__(self, parameter: dict[str, Any]) -> None:
        self._raw_dict = parameter
        self.name = parameter['name']
        self.type = get_type(parameter)
        self.log = parameter.get('log', False)
        # self.lower = None
        # self.upper = None
        # self.choices = None
        # self.sequence = None
        self.initial = None
        self.q = None
        self.step = None

        if 'log' in parameter:
            self.log = parameter['log']

        # if 'upper' in parameter:
        #     self.upper = parameter['lower']

        # if 'upper' in parameter:
        #     self.upper = parameter['upper']

        # if 'choices' in parameter:
        #     self.choices = parameter['choices']

        # if 'sequence' in parameter:
        #     self.sequence = parameter['sequence']

        self.lower = parameter.get('lower', None)
        self.upper = parameter.get('upper', None)
        self.choices = parameter.get('choices', None)
        self.sequence = parameter.get('sequence', None)

        if 'initial' in parameter:
            self.initial = parameter['initial']

        if 'q' in parameter:
            self.q = parameter['q']

        if 'step' in parameter:
            self.step = parameter['step']

        if 'base' in parameter:
            self.base = parameter['base']

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
        elif self.type.lower() == 'int':
            value = rng.randint(self.lower, self.upper)
        elif self.type.lower() == 'float':
            value = rng.uniform(self.lower, self.upper)
        elif self.type.lower() == 'categorical':
            value = rng.choice(self.choices)
        elif self.type.lower() == 'ordinal':
            value = rng.choice(self.sequence)
        else:
            raise TypeError(
                f'Invalid hyper parameter type: {self.type}')

        return {'name': self.name, 'type': self.type, 'value': value}


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
            self.hps[hps['name']] = HyperParameter(hps)

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
            raise KeyError(f'The parameter name {name} does not exist.')

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
