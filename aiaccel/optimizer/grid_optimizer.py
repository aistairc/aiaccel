from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any

from omegaconf.dictconfig import DictConfig

from aiaccel.optimizer import AbstractOptimizer
from aiaccel.parameter import HyperParameter


def get_grid_options(
    parameter_name: str,
    config: DictConfig
) -> tuple[Any, bool, Any]:
    """Get options about grid search.

    Args:
        parameter_name (str): A parameter name to get its options.
        config (Config): A config object.

    Returns:
        tuple[int | None, bool, int | None]: The first one is a base of
        logarithm parameter. The second one is logarithm parameter or not.
        The third one is a step of the grid.

    Raises:
        KeyError: Causes when step is not specified.
    """
    base: float | None = None
    log: bool = False
    step: float | None = None

    grid_options = config.optimize.parameters

    for g in grid_options:
        if g['name'] == parameter_name:
            if 'step' not in g.keys():
                raise KeyError(
                    f'No grid option `step` for parameter: {parameter_name}'
                )
            if 'log' not in g.keys():
                raise KeyError(
                    f'No grid option `log` for parameter: {parameter_name}'
                )
            if 'base' not in g.keys():
                raise KeyError(
                    f'No grid option `base` for parameter: {parameter_name}'
                )

            step = float(g['step'])
            log = bool(g['log'])
            if log:
                base = int(g['base'])

            return base, log, step

    raise KeyError(f'Invalid parameter name: {parameter_name}')


def generate_grid_points(
    p: HyperParameter, config: DictConfig
) -> dict[str, Any]:
    """Make a list of all parameters for this grid.

    Args:
        p (HyperParameter): A hyper parameter object.
        config (DictConfig): A configuration object.

    Returns:
        dict[str, str | list[float, int, str]]: A dictionary including all grid
        parameters.

    Raises:
        TypeError: Causes when an invalid parameter type is set.
    """
    new_param = {
        'parameter_name': p.name,
        'type': p.type
    }

    if p.type.lower() in ['int', 'float']:
        base, log, step = get_grid_options(p.name, config)
        lower = p.lower
        upper = p.upper

        if log:
            lower_x = base ** lower
            upper_x = base ** upper
            step_x = base ** step
            x = lower_x
            new_param['parameters'] = []
            while x < upper_x or math.isclose(x, upper_x, abs_tol=1e-10):
                new_param['parameters'].append(x)
                x *= step_x
        else:
            n = int((upper - lower) / step) + 1
            new_param['parameters'] = [lower + i * step for i in range(0, n)]
        if p.type.lower() == 'int':
            new_param['parameters'] = [int(i) for i in new_param['parameters']]

    elif p.type.lower() == 'categorical':
        new_param['parameters'] = p.choices

    elif p.type.lower() == 'ordinal':
        new_param['parameters'] = p.sequence

    else:
        raise TypeError(f'Invalid parameter type: {p.type}')

    return new_param


class GridOptimizer(AbstractOptimizer):
    """An optimizer class with grid search algorithm.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        ready_params (list[dict]): A list of ready hyper parameters.
        generate_index (int): A number of generated hyper parameters.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.ready_params = []
        for param in self.params.get_parameter_list():
            self.ready_params.append(generate_grid_points(param, self.config))
        self.generate_index = 0

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        self.generate_index = (
            self.storage.get_num_ready() +
            self.storage.get_num_running() +
            self.storage.get_num_finished()
        )

    def get_parameter_index(self) -> list[int] | None:
        """Get a next parameter index.

        Returns:
            list[int] | None: It returns None if all parameters are
            already generated.
        """
        parameter_lengths = [len(i['parameters']) for i in self.ready_params]
        remain = self.generate_index
        max_index = reduce(mul, parameter_lengths)

        if self.generate_index >= max_index:
            self.logger.warning('All parameters were generated.')
            return None

        parameter_index = []
        div = [
            reduce(
                lambda x, y: x * y, parameter_lengths[i + 1:]
            ) for i in range(0, len(parameter_lengths) - 1)
        ]

        for i in range(0, len(parameter_lengths) - 1):
            d = int(remain / div[i])
            parameter_index.append(d)
            remain -= d * div[i]

        parameter_index.append(remain)
        self.generate_index += 1

        return parameter_index

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new
                parameters. None if all of parameters are generated.
        """
        parameter_index = self.get_parameter_index()

        if parameter_index is None:
            self.logger.info('Generated all of parameters.')
            self.all_parameters_generated = True
            return None

        new_params: list[Any] = []
        for param, index in zip(self.ready_params, parameter_index):
            new_params.append(
                {
                    'parameter_name': param['parameter_name'],
                    'type': param['type'],
                    'value': param['parameters'][index]
                }
            )
        return new_params

    def generate_initial_parameter(
        self
    ) -> list[dict[str, float | int | str]]:
        """Generates initial parameters.

        Grid search algorithm always ignores the initial values in
        configulation file even if given.

        Raises:
            ValueError: Causes when the parameter is not generated.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        for hyperparameter in self.params.get_parameter_list():
            if hyperparameter.initial is not None:
                self.logger.warning(
                    "Initial values cannot be specified for grid search. "
                    "The set initial value has been invalidated."
                )
                break
        generated_parameter = self.generate_parameter()
        if generated_parameter is None:
            self.logger.error('Initial parameter not generaged.')
            raise ValueError('Initial parameter not generated.')
        else:
            return generated_parameter
