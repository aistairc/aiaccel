from __future__ import annotations

from omegaconf import DictConfig

from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import AbstractOptimizer


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    Args:
        config (DictConfig): A DictConfig object which has contents of
            configuration file and command line options.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.params: ConvertedParameterConfiguration = ConvertedParameterConfiguration(
            self.params, convert_int=False, convert_choices=False, convert_sequence=False
        )

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Args:
            None

        Returns:
            list[dict[str, float | int | str]]: A list of created parameters.
        """
        new_params = []
        for param in self.params.sample(self._rng):
            new_params.append({"parameter_name": param["name"], "type": param["type"], "value": param["value"]})

        return self.params.to_original_repr(new_params)
