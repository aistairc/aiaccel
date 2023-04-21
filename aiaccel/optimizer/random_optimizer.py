from __future__ import annotations

from omegaconf import DictConfig

from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import AbstractOptimizer


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.params = ConvertedParameterConfiguration(
            self.params, convert_int=False, convert_choices=False, convert_sequence=False
        )

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of created parameters.
        """
        new_params = []
        for param in self.params.get_parameter_list():
            if param.type == "uniform_float":
                value = self._rng.uniform(param.lower, param.upper)
            elif param.type == "uniform_int":
                value = self._rng.randint(param.lower, param.upper)
            elif param.type == "categorical":
                value = self._rng.choice(param.choices)
            elif param.type == "ordinal":
                value = self._rng.choice(param.sequence)
            else:
                assert False, f"Invalid type: {param.type}"

            new_params.append({"parameter_name": param.name, "type": param.type, "value": value})

        return self.params.to_original_repr(new_params)
