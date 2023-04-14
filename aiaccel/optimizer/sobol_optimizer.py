from __future__ import annotations

from omegaconf.dictconfig import DictConfig
from typing import Any

from scipy.stats import qmc

from aiaccel.optimizer import AbstractOptimizer
from aiaccel.parameter_conversion import ConvertedHyperparameterConfiguration


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

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.generate_index = 0
        self.sampler: Any = None
        self.converted_parameters = ConvertedHyperparameterConfiguration(
            self.params.get_parameter_list()
        )

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

        finished = self.storage.trial.get_finished()
        self.generate_index = len(finished)

        if self.config.resume is None or self.config.resume <= 0:
            self.sampler = qmc.Sobol(
                d=len(self.params.get_parameter_list()),
                scramble=self.config.optimize.sobol_scramble,
                seed=self._rng
            )

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        self.generate_index += 1

        vec = self.sampler.random()[0]

        new_params = []
        for vec_i, param in zip(vec, self.converted_parameters.get_list()):
            internal_value = (param.upper - param.lower) * vec_i + param.lower
            new_params.append(
                {
                    "parameter_name": param.name,
                    "type": param.type,
                    "value": param.convert_to_external_value(internal_value)
                }
            )

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
