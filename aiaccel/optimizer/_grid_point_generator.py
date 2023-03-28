from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod
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
NumericType = Union[float, int, np.floating, np.integer]


class GridCondition(ABC):
    """An abstract class for grid condition of a parameter.

    Args:
        hyperparameter (HyperParameter): A hyperparameter object.
    """

    def __init__(self, hyperparameter: HyperParameter) -> None:
        self._choices: list[GridValueType] = []
        self._num_choices: int = 0
        self._max_num_choices: int = sys.maxsize
        self.create_choices(hyperparameter)

    @abstractmethod
    def create_choices(self, hyperparameter: HyperParameter) -> None: ...

    def __iter__(self) -> Iterator[GridValueType]:
        return iter(self._choices)

    def __len__(self) -> int:
        return self._num_choices

    def has_choices(self) -> bool:
        """Whether the object has available choices.

        Returns:
            bool: True if the objecti has available choices.
        """
        return bool(self._choices)

    def has_num_choices(self) -> bool:
        """Whether the nonzero number of choices is set.

        Returns:
            bool: True if the number of choices is not zero.
        """
        return self._num_choices != 0

    def num_choices_incrementable(self) -> bool:
        """Whether the number of choices can be incremented.

        Returns:
            bool: True if the number of choices can be incremented.
        """
        return self.num_choices < self._max_num_choices

    @property
    def choices(self) -> list[GridValueType]:
        return self._choices

    @choices.setter
    def choices(self, choices: list[GridValueType]) -> None:
        self._choices = choices

    @property
    def num_choices(self) -> int:
        return self._num_choices

    @num_choices.setter
    def num_choices(self, num_choices: int) -> None:
        self._num_choices = num_choices


class FloatGridCondition(GridCondition):
    """A grid condition container for a float parameter.

    Args:
        hyperparameter (HyperParameter): A hyperparameter object.
    """

    def __init__(self, hyperparameter: HyperParameter) -> None:
        super().__init__(hyperparameter)

    def create_choices(self, hyperparameter: HyperParameter) -> None:
        """Creates choices from specified hyperparameter.

        The number of choices is specified by a property `num_choices` if it is
        changed from the default value of zero.
        Otherwise, `hyperparameter.num_numeric_choices` is used to specify the
        number of choices.

        Args:
            hyperparameter (HyperParameter): A hyperparameter object.
        """
        start = hyperparameter.lower
        stop = hyperparameter.upper
        if not self.has_num_choices():
            self.num_choices = hyperparameter.num_numeric_choices
        if hyperparameter.log:
            self.choices = list(map(float, np.geomspace(start, stop, self.num_choices)))
        else:
            self.choices = list(map(float, np.linspace(start, stop, self.num_choices)))

    @property
    def num_choices(self) -> int:
        return self._num_choices

    @num_choices.setter
    def num_choices(self, num_choices: int) -> None:
        if isinstance(num_choices, numeric_types):
            self._num_choices = max(int(num_choices), 0)
        else:
            self._num_choices = 0


def _cast_start_to_integer(start: NumericType) -> int:
    if start < 0:
        start = int(start)
    else:
        start = int(start) if np.isclose(int(start), start) else int(start) + 1
    return start


def _cast_stop_to_integer(stop: NumericType) -> int:
    if stop < 0:
        stop = int(stop) if np.isclose(int(stop), stop) else int(stop) - 1
    else:
        stop = int(stop)
    return stop


class IntGridCondition(GridCondition):
    """A grid condition container for an int parameter.

    Args:
        hyperparameter (HyperParameter): A hyperparameter object.
    """

    def __init__(self, hyperparameter: HyperParameter) -> None:
        super().__init__(hyperparameter)

    def create_choices(self, hyperparameter: HyperParameter) -> None:
        """Creates choices from specified hyperparameter.

        The number of choices is specified by a property `num_choices` if it is
        changed from the default value of zero.
        Otherwise, `hyperparameter.num_numeric_choices` is used to specify the
        number of choices.

        Args:
            hyperparameter (HyperParameter): A hyperparameter object.

        Raises:
            ValueError: Causes when lower and upper values of hyperparameter
                are rounded to the same integer.
        """
        if int(hyperparameter.lower) == int(hyperparameter.upper):
            raise ValueError(
                f'Invalid range of int parameter "{hyperparameter.name}": '
                f'{hyperparameter.lower} and {hyperparameter.upper} are rounded '
                f'to the same integer {int(hyperparameter.lower)}.'
            )
        start, stop = sorted([hyperparameter.lower, hyperparameter.upper])
        start = _cast_start_to_integer(start)
        stop = _cast_stop_to_integer(stop)
        self._max_num_choices = stop - start + 1
        if not self.has_num_choices():
            self.num_choices = hyperparameter.num_numeric_choices
        if hyperparameter.log:
            self.choices = sorted(set(map(int, np.geomspace(start, stop, self.num_choices))))
        else:
            self.choices = sorted(set(map(int, np.linspace(start, stop, self.num_choices))))

    @property
    def choices(self) -> list[GridValueType]:
        return self._choices

    @choices.setter
    def choices(self, choices: list[GridValueType]) -> None:
        self._choices = choices
        self.num_choices = len(choices)

    @property
    def num_choices(self) -> int:
        return self._num_choices

    @num_choices.setter
    def num_choices(self, num_choices: int) -> None:
        if isinstance(num_choices, numeric_types):
            self._num_choices = min(max(int(num_choices), 0), self._max_num_choices)
        else:
            self._num_choices = 0


class CategoricalGridCondition(GridCondition):
    """A grid condition container for a categorical parameter.

    Args:
        hyperparameter (HyperParameter): A hyperparameter object.
    """

    def __init__(self, hyperparameter: HyperParameter) -> None:
        super().__init__(hyperparameter)

    def create_choices(self, hyperparameter: HyperParameter) -> None:
        """Creates choices from specified hyperparameter.

        The choices are equivalent to `hyperparameter.sequence`.

        Args:
            hyperparameter (HyperParameter): A hyperparameter object.
        """
        self.choices = hyperparameter.choices
        self.num_choices = len(self.choices)
        self._max_num_choices = self.num_choices


class OrdinalGridCondition(GridCondition):
    """A grid condition container for an ordinal parameter.

    Args:
        hyperparameter (HyperParameter): A hyperparameter object.
    """

    def __init__(self, hyperparameter: HyperParameter) -> None:
        super().__init__(hyperparameter)

    def create_choices(self, hyperparameter: HyperParameter) -> None:
        """Creates choices from specified hyperparameter.

        The choices are equivalent to `hyperparameter.choices`.

        Args:
            hyperparameter (HyperParameter): A hyperparameter object.
        """
        self.choices = hyperparameter.sequence
        self.num_choices = len(self.choices)
        self._max_num_choices = self.num_choices


def _create_grid_condition(hyperparameter: HyperParameter) -> GridCondition:
    if hyperparameter.type == 'FLOAT':
        return FloatGridCondition(hyperparameter)
    elif hyperparameter.type == 'INT':
        return IntGridCondition(hyperparameter)
    elif hyperparameter.type == 'CATEGORICAL':
        return CategoricalGridCondition(hyperparameter)
    elif hyperparameter.type == 'ORDINAL':
        return OrdinalGridCondition(hyperparameter)
    else:
        raise TypeError(
            f'Specified parameter type "{hyperparameter.type}" of "{hyperparameter.name}" is invalid.'
        )


class GridConditionCollection:
    """Collection of GridCondition objects.

    Args:
        num_trials (int): The number of trials.
        hyperparameters (list[HyperParameter]): A list of Hyperparameter
            objects.
    """

    def __init__(self, num_trials: int, hyperparameters: list[HyperParameter]) -> None:
        self._num_trials = num_trials
        self._conditions: list[GridCondition] = []
        self._least_space_size = 1
        self._num_unspecified_parameters = 0
        self._residual_space_size = float(self._num_trials)
        self._num_larger_choices = 1
        self._num_larger_choice_parameters = self._num_unspecified_parameters
        self._num_smaller_choices = 0
        self._num_smaller_choice_parameters = 0
        self._register_grid_conditions(hyperparameters)

    def __contains__(self, value: object) -> bool:
        return value in self._conditions

    def __iter__(self) -> Iterator[GridCondition]:
        return iter(self._conditions)

    def __len__(self) -> int:
        return len(self._conditions)

    def _register_grid_conditions(self, hyperparameters: list[HyperParameter]) -> None:
        if hyperparameters:
            hyperparameter = hyperparameters.pop()
            grid_condition = _create_grid_condition(hyperparameter)
            self._conditions.append(grid_condition)
            self._register_grid_conditions(hyperparameters)
            if grid_condition.has_choices():
                self._least_space_size *= grid_condition.num_choices
            else:
                grid_condition.create_choices(hyperparameter)
        else:
            if grid_conditions_with_empty_choices := self._get_grid_conditions_with_empty_choices():
                self._set_num_choices(grid_conditions_with_empty_choices)

    def _get_grid_conditions_with_empty_choices(self) -> list[GridCondition]:
        empty_grid_conditions = []
        for grid_condition in self._conditions:
            if grid_condition.has_choices():
                continue
            empty_grid_conditions.append(grid_condition)
        return empty_grid_conditions

    def _set_num_choices(self, grid_conditions_with_empty_choices: list[GridCondition]) -> None:
        grid_condition = grid_conditions_with_empty_choices.pop(0)
        grid_condition.num_choices += 1
        grid_conditions_with_empty_choices.append(grid_condition)
        grid_space_size = 1
        is_incrementable = False
        for grid_condition in self._conditions:
            grid_space_size *= grid_condition.num_choices
            is_incrementable |= grid_condition.num_choices_incrementable()
        if is_incrementable and grid_space_size < self._num_trials:
            self._set_num_choices(grid_conditions_with_empty_choices)

    @ property
    def choices(self) -> list[list[GridValueType]]:
        return [grid_condition.choices for grid_condition in self._conditions]

    @ property
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
                    'To proceed, set "optimize.grid_accept_small_trial_number" `True` in config.'
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
