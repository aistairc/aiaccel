from __future__ import annotations

from scipy.stats import qmc

from aiaccel.optimizer import AbstractOptimizer


class SobolOptimizer(AbstractOptimizer):
    """An optimizer class with sobol algorithm.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        generate_index (int): A number of generated hyper parameters.
        sampler (Sobol): Engine for generating (scrambled) Sobol' sequences.

    Todo:
        Make it clear to resume this optimizer with Sobol sampler. Currentcode
        resume the sampler counts with a number of FINISHED PARAMETER FILES.
        Confirm whether the current code resumes for any timings of quits.
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        super().__init__(options)
        self.generate_index = None
        self.sampler = None

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        finished = self.storage.trial.get_finished()
        self.generate_index = len(finished)

        if self.options['resume'] is None or self.options['resume'] <= 0:
            self.sampler = qmc.Sobol(
                d=len(self.params.get_parameter_list()),
                scramble=self.config.sobol_scramble.get(),
                seed=self._rng
            )

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        l_params = self.params.get_parameter_list()
        n_params = len(l_params)

        new_params = []
        vec = self.sampler.random()[0]

        if self.generate_index is None:
            self.generate_index = 1
        else:
            self.generate_index += 1

        for i in range(0, n_params):
            min_value = l_params[i].lower
            max_value = l_params[i].upper
            value = (max_value - min_value) * vec[i] + min_value
            new_param = {
                'parameter_name': l_params[i].name,
                'type': l_params[i].type,
                'value': value
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
                "Initial values cannot be specified for sobol."
                "The set initial value has been invalidated."
            )
        return self.generate_parameter()
