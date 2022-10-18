import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

import aiaccel
from aiaccel.config import Config
from aiaccel.util.filesystem import load_yaml


def get_best_parameter(files: List[Path], goal: str, dict_lock: Path) ->\
        Tuple[Union[float, None], Union[Path, None]]:
    """Get a best parameter in specified files.

    Args:
        files (List[Path]): A list of files to find a best.
        goal (str): Maximize or Minimize.
        dict_lock (Path): A directory to store lock files.

    Returns:
        Tuple[Union[float, None], Union[Path, None]]: A best result value and a
            file path. It returns None if a number of files is less than one.

    Raises:
        ValueError: Causes when an invalid goal is set.
    """

    if len(files) < 1:
        return None, None

    yml = load_yaml(files[0], dict_lock)
    best = float(yml['result'])
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


def get_grid_options(
    parameter_name: str,
    config: Config
) -> Tuple[Union[int, None], bool, Union[int, None]]:

    """Get options about grid search.

    Args:
        parameter_name (str): A parameter name to get its options.
        config (ConfileWrapper): A config object.

    Returns:
        Tuple[Union[int, None], bool, Union[int, None]]: The first one is a
            base of logarithm parameter. The second one is logarithm parameter
            or not. The third one is a step of the grid.

    Raises:
        KeyError: Causes when step is not specified.
    """
    base = None
    log = False
    step = None

    grid_options = config.hyperparameters.get()

    for g in grid_options:
        if g['name'] == parameter_name:
            if 'step' in g.keys():
                step = float(g['step'])
            else:
                step = None
            log = bool(g['log'])
            if log:
                base = int(g['base'])
            break

    if step is None:
        raise KeyError(f'No grid option for parameter: {parameter_name}')
    else:
        return base, log, step


def get_type(parameter: dict) -> str:
    """Get a type of a specified parameter.

    Args:
        parameter (dict): A parameter dictionary in a configuration file.

    Returns:
        str: A parameter type any of 'INT', 'FLOAT', 'CATEGORICAL' and
            'ORDINAL'.
    """
    if parameter['type'] == 'uniform_int':
        return 'INT'
    elif parameter['type'] == 'uniform_float':
        return 'FLOAT'
    elif parameter['type'] == 'categorical':
        return 'CATEGORICAL'
    elif parameter['type'] == 'ordinal':
        return 'ORDINAL'
    else:
        return parameter['type']


class HyperParameter(object):
    """
    A hyper parameter class.

    Attributes:
        _raw_dict (dict): A parameter dictionary in a configuration file.
        name (str): A parameter name.
        type (str): A parameter type any of 'INT', 'FLOAT', 'CATEGORICAL' and
            'ORDINAL'.
        log (bool): A parameter is logarithm or not.
        lower (Union[float, int]): A lower value of a parameter.
        upper (Union[float, int]): A upper value of a parameter.
        choices (List[float, int, str]): This is set as a list of a parameter,
            when a parameter type is 'CATEGORICAL'.
        sequence (List[float, int, str]): This is set as a list of a parameter,
            when a parameter type is 'ORDINAL'.
        initial (Union[float, int, str]): A initial value. If this is set, this
            value is evaluated at first run.
        q (Union[float, int]): A quantization factor.
    """

    def __init__(self, parameter):
        """
        Args:
            parameter (dict): A parameter dictionary in a configuration file.
        """
        self._raw_dict = parameter
        self.name = parameter['name']
        self.type = get_type(parameter)
        self.log = False
        self.lower = None
        self.upper = None
        self.choices = None
        self.sequence = None
        self.initial = None
        self.q = None
        self.step = None

        if 'log' in parameter:
            self.log = parameter['log']

        if 'lower' in parameter:
            self.lower = parameter['lower']

        if 'upper' in parameter:
            self.upper = parameter['upper']

        if 'choices' in parameter:
            self.choices = parameter['choices']

        if 'sequence' in parameter:
            self.sequence = parameter['sequence']

        if 'initial' in parameter:
            self.initial = parameter['initial']

        if 'q' in parameter:
            self.q = parameter['q']

        if 'step' in parameter:
            self.step = parameter['step']

        if 'base' in parameter:
            self.step = parameter['base']

    def sample(self, initial: bool = False) -> dict:
        """Sample a parameter.

        Args:
            initial (bool): This is set, when a initial value is required.

        Returns:
            dict: A parameter dictionary.

        Raises:
            TypeError: Causes when an invalid type is set.
        """
        if initial and self.initial is not None:
            value = self.initial
        elif self.type == 'INT':
            value = np.random.randint(self.lower, self.upper)
        elif self.type == 'FLOAT':
            value = np.random.uniform(self.lower, self.upper)
        elif self.type == 'CATEGORICAL':
            value = np.random.choice(self.choices)
        elif self.type == 'ORDINAL':
            value = np.random.choice(self.sequence)
        else:
            raise TypeError(
                f'Invalid hyper parameter type: {self.type}')

        return {'name': self.name, 'type': self.type, 'value': value}


class HyperParameterConfiguration(object):
    """A configuration of hyper parameters.

    Attributes:
        json_string (dict): A configuration dictionary of hyper parameters.
        hps (dict): Hyper parameters.
    """

    def __init__(self, json_string: dict) -> None:
        """__init__ method.

        Args:
            json_string (dict): A configuration dictionary of hyper parameters.
        """
        self.json_string = json_string
        self.hps = {}

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

    def get_parameter_list(self) -> List[HyperParameter]:
        """Get a list of hyper parameter objects.

        Returns:
            List[HyperParameter]: A list of hyper parameter objects.
        """
        return list(self.hps.values())

    def get_parameter_dict(self) -> dict:
        """Get a dictionary of hyper parameters.

        Returns:
            dict: A hyper parameter dictionary.
        """
        return self.hps

    def sample(self, initial: bool = False) -> List[dict]:
        """Sample a hyper parameters set.

        Args:
            initial (bool): This is set, when a initial value is required.

        Returns:
            dict: A hyper parameters set.
        """
        ret = []

        for name, value in self.hps.items():
            ret.append(value.sample(initial))

        return ret


def load_parameter(json_string: dict) -> HyperParameterConfiguration:
    """Load HyperParameterConfiguration object from a configuration file.

    Args:
        json_string (dict): A hyper parameter configuration.

    Returns:
        HyperParameterConfiguration: A hyper parameter configuration.
    """
    return HyperParameterConfiguration(json_string)
