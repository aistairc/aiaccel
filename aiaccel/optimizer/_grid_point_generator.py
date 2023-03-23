from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Union
from typing import Literal

import numpy as np
from numpy.random import RandomState

from aiaccel.parameter import HyperParameter


GridValueType = Union[float, int, str]
SamplingMethodType = Literal['IN_ORDER', 'UNIFORM', 'RANDOM', 'DUPLICATABLE_RANDOM']
numeric_types = (float, int, np.floating, np.integer)


def _make_numeric_choices(hyperparameter: HyperParameter, num_choices: int | None = None) -> list[GridValueType]:
    choices: list[GridValueType] = []
    start = hyperparameter.lower
    stop = hyperparameter.upper
    num = num_choices or int(hyperparameter.num_numeric_choices)
    if hyperparameter.type == 'FLOAT':
        dtype = float
    elif hyperparameter.type == 'INT':
        dtype = int
    else:
        raise TypeError(
            f'Invalid parameter type "{hyperparameter.type}". "FLOAT" or "INT" required.'
        )
    dtype = int if hyperparameter.type == 'INT' else float
    if hyperparameter.log:
        choices = np.geomspace(start, stop, num, dtype=dtype).tolist()
    else:
        choices = np.linspace(start, stop, num, dtype=dtype).tolist()
    return choices


class GridCondition:
    """Condition of grid point.

    Args:
        hyperparameter (HyperParameter): HyperParameter object.

    Attributes:
        name (str): Parameter name.
        choices (list[float | int | str]): Choices of value of the parameter.
        num_choices (int): The number of choices of value.

    Raises:
        TypeError: Occurs when the specified parameter type is invalid.
    """

    def __init__(self, hyperparameter: HyperParameter) -> None:
        self.name = hyperparameter.name
        if hyperparameter.type in ('FLOAT', 'INT'):
            if isinstance(hyperparameter.num_numeric_choices, numeric_types):
                self.choices = _make_numeric_choices(hyperparameter)
                self.num_choices = int(hyperparameter.num_numeric_choices)
            else:
                self.choices = []
                self.num_choices = 0
        elif hyperparameter.type == 'CATEGORICAL':
            self.choices = hyperparameter.choices
            self.num_choices = len(self.choices)
        elif hyperparameter.type == 'ORDINAL':
            self.choices = hyperparameter.sequence
            self.num_choices = len(self.choices)
        else:
            raise TypeError(
                f'Specified parameter type "{hyperparameter.type}" for "{self.name}" is invalid.'
            )

    def __iter__(self) -> Iterator[GridValueType]:
        return iter(self.choices)

    def __len__(self) -> int:
        return self.num_choices


class GridConditionCollection:
    """_summary_

    Args:
        num_trials (int): _description_
        hyperparameters (list[HyperParameter]): _description_

    Raises:
        ValueError: _description_
    """

    def __init__(self, num_trials: int, hyperparameters: list[HyperParameter]) -> None:
        self.num_trials = num_trials
        self._conditions: list[GridCondition] = []
        self._least_space_size = 1
        self._num_unspecified_parameters = 0
        self._residual_space_size = float(self.num_trials)
        self._num_larger_choices = 1
        self._num_larger_choice_parameters = self._num_unspecified_parameters
        self._num_smaller_choices = 0
        self._num_smaller_choice_parameters = 0
        not_used_num_choices = self._register_grid_conditions(hyperparameters)
        if not_used_num_choices:
            raise ValueError(
                'Failed to assign choice for numeric parameter.'
            )

    def __iter__(self) -> Iterator[GridCondition]:
        return iter(self._conditions)

    def __len__(self) -> int:
        return len(self._conditions)

    def _register_grid_conditions(self, hyperparameters: list[HyperParameter]) -> list[int]:
        if hyperparameters:
            hyperparameter = hyperparameters.pop()
            grid_condition = GridCondition(hyperparameter)
            self._update_least_space_size(grid_condition)
            self._update_num_unspecified_parameters(grid_condition)
            nums_choices = self._register_grid_conditions(hyperparameters)
            if grid_condition.num_choices == 0:
                num_choices = nums_choices.pop()
                grid_condition.num_choices = num_choices
                grid_condition.choices = _make_numeric_choices(hyperparameter, num_choices)
            self._conditions.append(grid_condition)
            return nums_choices
        else:
            self._update_residual_space_size()
            if self._num_unspecified_parameters:
                self._calc_num_choices()
                self._split_num_unspecified_parameters()
                return self._get_auto_defined_num_choices()
            else:
                return []

    def _update_least_space_size(self, grid_condition: GridCondition) -> None:
        self._least_space_size *= grid_condition.num_choices or 1

    def _update_num_unspecified_parameters(self, grid_condition: GridCondition) -> None:
        self._num_unspecified_parameters += int(grid_condition.num_choices == 0)
        self._num_larger_choice_parameters = self._num_unspecified_parameters

    def _update_residual_space_size(self) -> None:
        self._residual_space_size /= self._least_space_size

    def _calc_num_choices(self) -> None:
        if (
            self._num_larger_choices ** self._num_unspecified_parameters <
            self._residual_space_size
        ):
            self._num_larger_choices += 1
            self._num_smaller_choices += 1
            self._calc_num_choices()

    def _split_num_unspecified_parameters(self) -> None:
        if (
            self._num_larger_choices ** (self._num_larger_choice_parameters - 1) *
            self._num_smaller_choices ** (self._num_smaller_choice_parameters + 1) >
            self._residual_space_size
        ):
            self._num_larger_choice_parameters -= 1
            self._num_smaller_choice_parameters += 1
            self._split_num_unspecified_parameters()

    def _get_auto_defined_num_choices(self) -> list[int]:
        nums_choices = (
            [self._num_larger_choices] * self._num_larger_choice_parameters +
            [self._num_smaller_choices] * self._num_smaller_choice_parameters
        )
        return nums_choices

    @property
    def choices(self) -> list[list[GridValueType]]:
        return [grid_condition.choices for grid_condition in self._conditions]

    @property
    def least_space_size(self) -> int:
        return self._least_space_size


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
                - UNIFORM - Samples after thin out the grid points and thus
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
        num_trials: int,
        hyperparameters: list[HyperParameter],
        sampling_method: SamplingMethodType = 'IN_ORDER',
        rng: RandomState | None = None,
        accept_small_trial_number: bool = False
    ) -> None:
        self._num_trials = num_trials
        self._grid_condition_collection = GridConditionCollection(self._num_trials, hyperparameters)
        if self._num_trials < self._grid_condition_collection.least_space_size:
            if not accept_small_trial_number:
                raise ValueError(
                    f'Too small "trial_num": {self._num_trials} (required '
                    f'{self._grid_condition_collection.least_space_size} or greater). '
                    'To proceed, use "--accept-small-trial-number" option.'
                )
        self._parameter_lengths = list(map(len, self._grid_condition_collection))
        self._grid_space_size = int(np.prod(self._parameter_lengths))
        self._digits = np.cumprod(self._parameter_lengths[::-1])[-2::-1]
        self._sampling_method = sampling_method
        self._num_generated_points = 0

        if rng is None:
            self._rng = RandomState(None)
        else:
            self._rng = rng

        if self._sampling_method == 'RANDOM':
            self._grid_point_stack = list(product(*self._grid_condition_collection))
        else:
            self._grid_point_stack = []

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
        elif self._sampling_method == 'UNIFORM':
            next_grid_point = self._get_grid_point_uniformly(
                self._num_generated_points)
        elif self._sampling_method == 'RANDOM':
            next_grid_point = self._get_grid_point_randomly(
                self._num_generated_points)
        elif self._sampling_method == 'DUPLICATABLE_RANDOM':
            next_grid_point = self._get_grid_point_duplicatable_randomly(
                self._num_generated_points
            )
        else:
            raise ValueError(
                f'Invalid sampling method: {self._sampling_method}'
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
        for index, param in zip(indices, self._grid_condition_collection):
            next_grid_point.append(param.choices[index])
        return next_grid_point

    def _get_grid_point_uniformly(self, trial_id: int) -> list[GridValueType]:
        converted_trial_id = int(np.linspace(
            0, self._grid_space_size - 1, self._num_trials, dtype=int
        )[trial_id])
        return self._get_grid_point_in_order(converted_trial_id)

    def _get_grid_point_randomly(self, trial_id: int) -> list[GridValueType]:
        index = self._rng.randint(0, len(self._grid_point_stack))
        next_grid_point = self._grid_point_stack.pop(index)
        return list(next_grid_point)

    def _get_grid_point_duplicatable_randomly(self, trial_id: int) -> list[GridValueType]:
        index = self._rng.randint(0, self._grid_space_size)
        return self._get_grid_point_in_order(index)

    @ property
    def num_generated_points(self) -> int:
        return self._num_generated_points
