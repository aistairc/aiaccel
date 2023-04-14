from __future__ import annotations

from omegaconf import DictConfig

from aiaccel.optimizer import AbstractOptimizer
from aiaccel.parameter_conversion import ConvertedHyperparameterConfiguration


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.converted_parameters = ConvertedHyperparameterConfiguration(
            self.params
        )

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of created parameters.
        """
        new_params = []
        for param in self.converted_parameters.converted_parameters.values():
            if param.type == "uniform_float":
                internal_value = self._rng.uniform(param.lower, param.upper)
            elif param.type == "uniform_int":
                internal_value = self._rng.uniform(param.lower, param.upper)
            elif param.type == "categorical":
                internal_value = self._rng.choice(param.choices)
            elif param.type == "ordinal":
                internal_value = self._rng.choice(param.sequence)
            else:
                assert False, f"Invalid type: {param.type}"

            new_params.append(
                {
                    "parameter_name": param.name,
                    "type": param.type,
                    "value": param.convert_to_external_value(internal_value)
                }
            )

        return new_params
