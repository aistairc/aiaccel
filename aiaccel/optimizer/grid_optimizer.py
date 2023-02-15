from __future__ import annotations
import math
from functools import reduce
from operator import mul

from aiaccel.config import Config
from aiaccel.optimizer import AbstractOptimizer
from aiaccel.parameter import HyperParameter


def get_grid_options(
    parameter_name: str,
    config: Config
) -> tuple[int | None, bool, int | None]:
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

    grid_options = config.hyperparameters.get()

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
    p: HyperParameter, config: Config
) -> dict[str, float | int | str | list[float, int, str]]:
    """Make a list of all parameters for this grid.

    Args:
        p (HyperParameter): A hyper parameter object.
        config (Config): A config object.

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
        new_param['parameters'] = list(p.choices)

    elif p.type.lower() == 'ordinal':
        new_param['parameters'] = list(p.sequence)

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

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        super().__init__(options)
        self.ready_params = None
        self.generate_index = None

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        self.ready_params = []

        for param in self.params.get_parameter_list():
            self.ready_params.append(generate_grid_points(param, self.config))

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
                lambda x, y: x * y, parameter_lengths[0:-1 - i]
            ) for i in range(0, len(parameter_lengths) - 1)
        ]

        for i in range(0, len(parameter_lengths) - 1):
            d = int(remain / div[i])
            parameter_index.append(d)
            remain -= d * div[i]

        parameter_index.append(remain)
        self.generate_index += 1

        return parameter_index

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        parameter_index = self.get_parameter_index()
        new_params = []

        if parameter_index is None:
            self.logger.info('Generated all of parameters.')
            self.all_parameter_generated = True
            return new_params

        for i in range(0, len(self.ready_params)):
            new_param = {
                'parameter_name': self.ready_params[i]['parameter_name'],
                'type': self.ready_params[i]['type'],
                'value': self.ready_params[i]['parameters']
                [parameter_index[i]]
            }
            new_params.append(new_param)

        return new_params

    def generate_initial_parameter(
        self
    ) -> list[dict[str, float | int | str]]:
        """Generate initial parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        if super().generate_initial_parameter() is not None:
            self.logger.warning(
                "Initial values cannot be specified for grid search."
                "The set initial value has been invalidated."
            )
        return self.generate_parameter()
