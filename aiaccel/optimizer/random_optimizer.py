from __future__ import annotations

from omegaconf.dictconfig import DictConfig

from aiaccel.config import is_multi_objective
from aiaccel.optimizer import AbstractOptimizer


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        if is_multi_objective(self.config):
            raise NotImplementedError("Random optimizer does not support multi-objective " "optimization.")

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of created parameters.
        """
        new_params = []
        sample = self.params.sample(rng=self._rng)

        for s in sample:
            new_param = {"parameter_name": s["name"], "type": s["type"], "value": s["value"]}
            new_params.append(new_param)

        return new_params
