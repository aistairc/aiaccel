from __future__ import annotations

from typing import Any


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

    def sample(self, initial: bool = False, rng: Any = None) -> dict[str, Any]:
        """Sample a parameter.

        Args:
            initial (bool): This is set, when a initial value is required.
            rng (np.random.RandomState): A reference to a random generator.

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

    def sample(self, initial: bool = False, rng: Any = None
               ) -> list[dict[str, Any]]:
        """Sample a hyper parameters set.

        Args:
            initial (bool, optional): This is set, when a initial value is
                required.
            rng (np.random.RandomState): A reference to a random generator.

        Returns:
            list[dict]: A hyper parameters set.
        """
        ret = []
        for name, value in self.hps.items():
            ret.append(value.sample(initial, rng=rng))
        return ret


def load_parameter(json_string: dict[str, Any]) -> HyperParameterConfiguration:
    """Load HyperParameterConfiguration object from a configuration file.

    Args:
        json_string (dict): A hyper parameter configuration.

    Returns:
        HyperParameterConfiguration: A hyper parameter configuration.
    """
    return HyperParameterConfiguration(json_string)
