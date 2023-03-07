from __future__ import annotations

from aiaccel.config import Config
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer._grid_point_generator import GridPointGenerator


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

        option_accept_small_trial_number = options.get('accept_small_trial_number', False)
        config_accept_small_trial_number = self.config.grid_accept_small_trial_number.get()
        accept_small_trial_number = option_accept_small_trial_number or config_accept_small_trial_number

        if accept_small_trial_number:
            self.logger.warning(
                'The option "accept_small_trial_number" is valid. '
                'Some of grid points may be not seached.'
            )

        try:
            self._grid_point_generator = GridPointGenerator(
                self.params.get_parameter_list(),
                self.config.trial_number.get(),
                sampling_method=self.config.grid_sampling_method.get().upper(),
                rng=self._rng,
                accept_small_trial_number=accept_small_trial_number
            )
        except ValueError as exception:
            self.logger.error(exception)
            raise exception

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generates parameters.

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
<< << << < HEAD
                lambda x, y: x * y, parameter_lengths[0:-1 - i]
== == == =
                lambda x, y: x * y, parameter_lengths[i + 1:]
>> >>>> > 8d45583(separates grid optimizers)
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


<< << << < HEAD
            self.all_parameter_generated = True
            return new_params
== == == =
            if self.storage.get_num_finished() >= self.generate_index:
                self.all_parameters_generated = True
            return new_params

        for i in range(0, len(self.ready_params)):
            new_param = {
                'parameter_name': self.ready_params[i]['parameter_name'],
                'type': self.ready_params[i]['type'],
                'value': self.ready_params[i]['parameters']
                [parameter_index[i]]
            }
            new_params.append(new_param)
>>>>>>> 8d45583 (separates grid optimizers)

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
