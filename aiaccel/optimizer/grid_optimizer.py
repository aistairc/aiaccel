from __future__ import annotations

from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer._grid_point_generator import GridPointGenerator


class GridOptimizer(AbstractOptimizer):
    """An optimizer class with grid search algorithm.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing command line options.

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
            list[dict[str, float | int | str]] | None: A list of new parameters.
        """
        if self.all_parameters_generated:
            return None

        if self._grid_point_generator.all_grid_points_generated():
            self.logger.info('Generated all of parameters.')
            self.all_parameters_generated = True
            return None

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
