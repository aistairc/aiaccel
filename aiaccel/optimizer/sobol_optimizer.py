from __future__ import annotations

from typing import Any

from omegaconf.dictconfig import DictConfig
from scipy.stats import qmc

from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import AbstractOptimizer


class SobolOptimizer(AbstractOptimizer):
    """An optimizer class with sobol algorithm.

    Args:
        config (DictConfig): A DictConfig object which has contents of
            configuration file and command line options.

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
        self.params: ConvertedParameterConfiguration = ConvertedParameterConfiguration(self.params)

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
                d=len(self.params.get_parameter_list()), scramble=self.config.optimize.sobol_scramble, seed=self._rng
            )

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        self.generate_index += 1

        vec = self.sampler.random()[0]

        new_params = []
        for vec_i, param in zip(vec, self.params.get_parameter_list()):
            value = (param.upper - param.lower) * vec_i + param.lower
            new_params.append({"parameter_name": param.name, "type": param.type, "value": value})

        return self.params.to_original_repr(new_params)

    def generate_initial_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate initial parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        if super().generate_initial_parameter() is not None:
            self.logger.warning(
                "Initial values cannot be specified for sobol." "The set initial value has been invalidated."
            )
        return self.generate_parameter()
