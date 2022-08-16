from aiaccel.config import Config
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.parameter import HyperParameter, get_grid_options
from aiaccel.util.filesystem import get_file_hp_finished, get_file_hp_ready,\
    get_file_hp_running
from functools import reduce
from operator import mul
from typing import List, Optional, Union
import math


def generate_grid_points(p: HyperParameter, config: Config) -> dict:
    """Make a list of all parameters for this grid.

    Args:
        p (HyperParameter): A hyper parameter object.
        config (ConfileWrapper): A configuration object.

    Returns:
        dict: A dictionary including all grid parameters.

    Raises:
        TypeError: Causes when an invalid parameter type is set.
    """
    new_param = {
        'parameter_name': p.name,
        'type': p.type
    }

    if p.type in ['INT', 'FLOAT']:
        base, log, step = get_grid_options(p.name, config)
        lower = p.lower
        upper = p.upper
        n = int((upper - lower) / step) + 1

        if log:
            lower_x = lower ** base
            upper_x = upper ** base
            x = lower_x
            new_param['parameters'] = []

            while x < upper_x or math.isclose(x, upper_x, abs_tol=1e-10):
                new_param['parameters'].append(math.log(x, base))
                x += step

            new_param['parameters'].append(upper)
        else:
            new_param['parameters'] = [lower + i * step for i in range(0, n)]

        if p.type == 'INT':
            new_param['parameters'] = [int(i) for i in new_param['parameters']]

    elif p.type == 'CATEGORICAL':
        new_param['parameters'] = list(p.choices)

    elif p.type == 'ORDINAL':
        new_param['parameters'] = list(p.sequence)

    else:
        raise TypeError('Invalid parameter type: {}'.format(p.type))

    return new_param


class GridSearchOptimizer(AbstractOptimizer):
    """An optimizer class with grid search algorithm.

    Attributes:
        ready_params (List[dict]): A list of ready hyper parameters.
        generate_index (int): A number of generated hyper parameters.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of Optimizer.

        Args:
            config (str): A file name of a configuration.
        """
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

        ready_files = get_file_hp_ready(self.ws, self.dict_lock)
        running_files = get_file_hp_running(self.ws, self.dict_lock)
        finished_files = get_file_hp_finished(self.ws, self.dict_lock)

        self.generate_index = len(ready_files) + len(running_files) +\
            len(finished_files)

    def get_parameter_index(self) -> Union[List[int], None]:
        """Get a next parameter index.

        Returns:
            Union[List[int], None]: It returns None if all parameters are
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
                lambda x, y: x * y,
                parameter_lengths[0:-1 - i]
            ) for i in range(0, len(parameter_lengths) - 1)
        ]

        for i in range(0, len(parameter_lengths) - 1):
            d = int(remain / div[i])
            parameter_index.append(d)
            remain -= d * div[i]

        parameter_index.append(remain)
        self.generate_index += 1

        return parameter_index

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """
        returned_params = []

        for n in range(number):
            parameter_index = self.get_parameter_index()
            new_params = []

            if parameter_index is None:
                self.logger.info('Generated all of parameters.')
                self.all_parameter_generated = True
                break

            for i in range(0, len(self.ready_params)):
                new_param = {
                    'parameter_name': self.ready_params[i]['parameter_name'],
                    'type': self.ready_params[i]['type'],
                    'value': self.ready_params[i]['parameters']
                    [parameter_index[i]]
                }
                new_params.append(new_param)

            returned_params.append({'parameters': new_params})
            self.generated_parameter += 1

        self.create_parameter_files(returned_params)

    def _serialize(self) -> None:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """
        self.serialize_datas = {
            'generated_parameter': self.generated_parameter,
            'loop_count': self.loop_count,
            'ready_params': self.ready_params,
            'generate_index': self.generate_index
        }
        return super()._serialize()

    def _deserialize(self, dict_objects: dict) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(dict_objects)
        self.ready_params = dict_objects['ready_params']
        self.generate_index = dict_objects['generate_index']
