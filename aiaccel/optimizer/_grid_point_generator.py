from __future__ import annotations

from itertools import product
from typing import Union
from typing import Literal

import numpy as np
from numpy.random import RandomState

from aiaccel.parameter import HyperParameter


GridValueType = Union[float, int, str]
SamplingMethodType = Literal['IN_ORDER', 'THIN_OUT', 'RANDOM', 'DUPLICATABLE_RANDOM']


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


def _suggest_nums_grid_points(
    grid_space_size: int,
    least_grid_space_size: int,
    num_parameter: int,
) -> list[int]:
    if num_parameter:
        upper_points = 1
        while (
            least_grid_space_size * upper_points ** num_parameter < grid_space_size
        ):
            upper_points += 1
        lower_points = upper_points - 1

        num_lower_points_parameter = 0
        num_upper_points_parameter = num_parameter
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
    else:
        return []


def _generate_all_grid_points(
    hyperparameters: list[HyperParameter],
    suggested_nums_grid_points: list[int]
) -> list[list[GridValueType]]:
    grid_points: list[list[GridValueType]] = []
    for hyperparameter in hyperparameters:
        if hyperparameter.type in ('INT', 'FLOAT'):
            if isinstance(hyperparameter.num_grid_points, int):
                num = hyperparameter.num_grid_points
            else:
                num = suggested_nums_grid_points.pop()
            if hyperparameter.log:
                if hyperparameter.type == 'INT':
                    grid_points.append(
                        np.geomspace(hyperparameter.lower,
                                     hyperparameter.upper, num,
                                     dtype=int).tolist()
                    )
                else:
                    grid_points.append(
                        np.geomspace(hyperparameter.lower,
                                     hyperparameter.upper, num,
                                     dtype=float).tolist()
                    )
            else:
                if hyperparameter.type == 'INT':
                    grid_points.append(
                        np.linspace(hyperparameter.lower,
                                    hyperparameter.upper, num,
                                    dtype=int).tolist()
                    )
                else:
                    grid_points.append(
                        np.linspace(hyperparameter.lower,
                                    hyperparameter.upper, num,
                                    dtype=float).tolist()
                    )
        elif hyperparameter.type == 'CATEGORICAL':
            grid_points.append(hyperparameter.choices)
        elif hyperparameter.type == 'ORDINAL':
            grid_points.append(hyperparameter.sequence)
    return grid_points


class GridPointGenerator:
    """Generator of grid points.

    Args:
        hyperparameters (list[HyperParameter]): A list of HyperParameter
            object.
        trial_number (int): The number of trials.
        sampling_method (SamplingMethod, optional): Method to sample grid
            points. In the case the trial number matches the number of
            generated grid points, this option basically affect only seach
            order. On the other hand, the trial number does not match the
            number of grid points results in ignoring of some grid points which
            are determined by this option. Available options are as follows;
                - IN_ORDER (default) - Samples in order. If trial number is
                smaller than the number of generated grid points, the grid
                points near the edge of search space may be ignored.
                - THIN_OUT - Samples after thin out the grid points and thus
                the ignored points does not gather cirtein region.
                - RANDOM - Samples randomly.
                - DUPLICATABLE_RANDOM - Samples randomly but choice duplication
                may occur. This option should be used only for the numerous
                grid points such that memory error occurs when using RANDOM.
                Although RANDOM stacks combination of grid points, this option
                stacks grid points only for each parameter. This saves memory
                in exchange for possible duplication.

            Defaults to SamplingMethod.IN_ORDER.
        rng (RandomState | None, optional): RandomState. If None, the
            constructor makes a random generator by `RandomState(None)`.
            Defaults to None.
        accept_small_trial_number (bool, optional): Whether to accept smaller
            trial number than the least grid space size. If True, use the given
            trial number while ignoring some grid points. Defaults to False.

    Raises:
        Warning: Causes when trial_num is smaller than the number of grid
            point in the least grid space composed of parameters with fixed
            grid points.
        ValueError: Causes when trial_num is smaller than the number of grid
            point in the least grid space composed of parameters with fixed
            grid points.

    """

    def __init__(
        self,
        hyperparameters: list[HyperParameter],
        trial_number: int,
        sampling_method: SamplingMethodType = 'IN_ORDER',
        rng: RandomState | None = None,
        accept_small_trial_number: bool = False
    ) -> None:
        nums_fixed_grid_points = _count_fixed_grid_points(hyperparameters)
        least_grid_space_size = int(np.prod(
            nums_fixed_grid_points, where=np.array(nums_fixed_grid_points) != 0
        ))

        if trial_number < least_grid_space_size:
            if not accept_small_trial_number:
                raise ValueError(
                    f'Too small "trial_num": {trial_number} (required '
                    f'{least_grid_space_size} or greater). '
                    'To proceed, use "--accept-small-trial-number" option.'
                )

        suggested_nums_grid_points = _suggest_nums_grid_points(
            grid_space_size=trial_number,
            least_grid_space_size=least_grid_space_size,
            num_parameter=nums_fixed_grid_points.count(0),
        )

        self._point_list = _generate_all_grid_points(
            hyperparameters, suggested_nums_grid_points
        )
        self._parameter_lengths = list(map(len, self._point_list))
        self._grid_space_size = int(np.prod(self._parameter_lengths))
        self._digits = np.cumprod(self._parameter_lengths[::-1])[-2::-1]
        self._trial_number = trial_number
        self._sampling_method = sampling_method
        self._num_generated_points = 0

        if (
            self._sampling_method == 'RANDOM' or
            self._sampling_method == 'DUPLICATABLE_RANDOM'
        ):
            if rng is None:
                self._rng = RandomState(None)
            else:
                self._rng = rng
            self._grid_point_stack = list(product(*self._point_list))
            self._generated_grid_point_stack: list[tuple[GridValueType]] = []

    def all_grid_points_generated(self) -> bool:
        """Whether all grid points are generated.

        Returns:
            bool: True if all grid point are generated.
        """
        return self._num_generated_points >= self._grid_space_size

    def get_next_grid_point(self) -> list[GridValueType]:
        """Gets a next parameter.

        Returns:
            list[GridValueType]: A list of parameters.
        """
        if self._sampling_method == 'IN_ORDER':
            next_grid_point = self._get_grid_point_in_order(
                self._num_generated_points)
        elif self._sampling_method == 'THIN_OUT':
            next_grid_point = self._get_grid_point_thin_out(
                self._num_generated_points)
        elif self._sampling_method == 'RANDOM':
            next_grid_point = self._get_grid_point_random(
                self._num_generated_points)
        else:  # self._sampling_method == 'DUPLICATABLE_RANDOM':
            next_grid_point = self._get_grid_point_duplicatable_random(
                self._num_generated_points
            )
        self._num_generated_points += 1
        return next_grid_point

    def _get_grid_point_in_order(self, trial_id: int) -> list[GridValueType]:
        remain = trial_id
        indices = []
        for i in self._digits:
            indices.append(remain // i)
            remain = trial_id % i
        indices.append(remain)
        next_grid_point = []
        for index, param in zip(indices, self._point_list):
            next_grid_point.append(param[index])
        return next_grid_point

    def _get_grid_point_thin_out(self, trial_id: int) -> list[GridValueType]:
        converted_trial_id = int(np.linspace(
            0, self._grid_space_size - 1, self._trial_number, dtype=int
        )[trial_id])
        return self._get_grid_point_in_order(converted_trial_id)

    def _get_grid_point_random(self, trial_id: int) -> list[GridValueType]:
        index = self._rng.randint(0, len(self._grid_point_stack))
        next_grid_point = self._grid_point_stack.pop(index)
        return list(next_grid_point)

    def _get_grid_point_duplicatable_random(self, trial_id: int) -> list[GridValueType]:
        index = self._rng.randint(0, self._grid_space_size)
        return self._get_grid_point_in_order(index)

    @ property
    def num_generated_points(self) -> int:
        return self._num_generated_points
