from __future__ import annotations
import math
from functools import reduce
from operator import mul
from typing import Any

import numpy as np

from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.parameter import HyperParameter


def _count_fixed_grid_points(hyperparameters: list[HyperParameter]) -> list[int]:
    nums_fixed_grid_points: list[int] = []
    for hyperparameter in hyperparameters:
        if hyperparameter.type in ('INT', 'FLOAT'):
            if isinstance(hyperparameter.num_grid_points, int):
                nums_fixed_grid_points.append(hyperparameter.num_grid_points)
            else:
                nums_fixed_grid_points.append(0)
        elif hyperparameter.type == 'CATEGORICAL':
            nums_fixed_grid_points.append(len(hyperparameter.choices))
        elif hyperparameter.type == 'ORDINAL':
            nums_fixed_grid_points.append(len(hyperparameter.sequence))
    return nums_fixed_grid_points


def _suggest_grid_points(
    grid_space_size: int,
    least_grid_space_size: int,
    num_point_free_parameter: int,
) -> list[int]:
    upper_points = 1
    while (
        least_grid_space_size *
        upper_points ** num_point_free_parameter < grid_space_size
    ):
        upper_points += 1
    lower_points = upper_points - 1

    num_lower_points_parameter = 0
    num_upper_points_parameter = num_point_free_parameter
    while (
        least_grid_space_size *
        lower_points ** (num_lower_points_parameter + 1) *
        upper_points ** (num_upper_points_parameter - 1) > grid_space_size
    ):
        num_lower_points_parameter += 1
        num_upper_points_parameter -= 1

    calculated_grid_points = (
        [lower_points] * num_lower_points_parameter +
        [upper_points] * num_upper_points_parameter
    )
    return calculated_grid_points


def _generate_all_grid_points(
    hyperparameters: list[HyperParameter], calculated_grid_points: list
) -> list[np.ndarray | list]:
    grid_points: list[np.ndarray | list] = []
    for hyperparameter in hyperparameters:
        if hyperparameter.type in ('INT', 'FLOAT'):
            if isinstance(hyperparameter.num_grid_points, int):
                num = hyperparameter.num_grid_points
            else:
                num = calculated_grid_points.pop()
            if hyperparameter.log:
                grid_points.append(
                    np.geomspace(hyperparameter.lower, hyperparameter.upper, num)
                )
            else:
                grid_points.append(
                    np.linspace(hyperparameter.lower, hyperparameter.upper, num)
                )
        elif hyperparameter.type == 'CATEGORICAL':
            grid_points.append(hyperparameter.choices)
        elif hyperparameter.type == 'ORDINAL':
            grid_points.append(hyperparameter.sequence)
    return grid_points


def _select_grid_points(point_lists: list, trial_number: int) -> list[Any]:
    num_generated_grid_point = len()
    selected_grid_point_ids = np.linspace(
        0, num_generated_grid_point - 1, trial_number, dtype=int
    )
    selected_grid_points = []
    for i in selected_grid_point_ids:
        selected_grid_points.append(grid_points[i])

    return selected_grid_points


class _GridPoints:
    def __init__(self, point_list: list[np.ndarray | list]) -> None:
        self._point_list = point_list

    def __call__(self, index: int) -> list[Any]:
        return _select_grid_points(self._point_list, index)


def _generate_grid_points(
    hyperparameters: list[HyperParameter],
    trial_number: int
) -> list:
    """Makes a list of grid points.

    Args:
        hyperparameters (list[HyperParameter]): A list of A hyper parameters.
        trial_number (int): The number of total trials.

    Returns:
        list: A list of grid points.

    Raises:
        ValueError: Causes when trial_num is smaller than the number of grid
            point in the least space composed of parameters with fixed grid
            values.
    """

    nums_fixed_grid_points = _count_fixed_grid_points(hyperparameters)
    least_grid_space_size = np.prod(
        nums_fixed_grid_points, where=np.array(nums_fixed_grid_points) != 0
    )

    if trial_number < least_grid_space_size:
        raise ValueError(
            'Too small "trial_num": Grid search can not be completed'
        )

    suggeted_grid_points = _suggest_grid_points(
        grid_space_size=trial_number,
        least_grid_space_size=least_grid_space_size,
        num_parameter=nums_fixed_grid_points.count(0),
    )

    point_list = _generate_all_grid_points(
        hyperparameters, suggeted_grid_points
    )

    # grid_points = _GridPoints(point_list)
    # return grid_points

    return point_list


class GridOptimizer(AbstractOptimizer):
    """An optimizer class with grid search algorithm.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing command line options.

    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        super().__init__(options)
        # self.ready_params = _generate_grid_points(
        #     self.params.get_parameter_list(), self.config.trial_number.get()
        # )
        self._grid_points = _generate_grid_points(
            self.params.get_parameter_list(), self.config.trial_number.get()
        )

        # self.generate_index = 0
        self._num_generated_grid_points = 0

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        # self.generate_index = (
        #     self.storage.get_num_ready() +
        #     self.storage.get_num_running() +
        #     self.storage.get_num_finished()
        # )
        self._num_generated_grid_points = (
            self.storage.get_num_ready() +
            self.storage.get_num_running() +
            self.storage.get_num_finished()
        )

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generates parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new parameters.
        """
        if self.all_parameters_generated:
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

        new_params: list[dict[str, float | int | str]] = []
        for param, value in zip(
                self.params.get_parameter_list(),
                self._grid_point_generator.get_next_grid_point()
        ):
            new_params.append(
                {
                    'parameter_name': param.name,
                    'type': param.type,
                    'value': value
                }
            )
        return new_params

    def generate_initial_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generates initial parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new parameters.
        """
        if super().generate_initial_parameter() is not None:
            self.logger.warning(
                "Initial values cannot be specified for grid search. "
                "The set initial value has been invalidated."
            )
        return self.generate_parameter()
