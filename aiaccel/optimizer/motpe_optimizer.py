from __future__ import annotations

from typing import Any

import optuna
from omegaconf.dictconfig import DictConfig

from aiaccel.optimizer.tpe_optimizer import (
    LazyRandomStateWrapper,
    RandomSamplerWrapper,
    TpeOptimizer,
    TPESamplerWrapper,
    create_distributions,
)


class MOTpeOptimizer(TpeOptimizer):
    """An optimizer class based on multi-objective optuna.samplers.TPESampler.

    Args:
        options (dict): A dictionary containing command line options.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.parameter_pool: dict[str, Any] = {}
        self.study_name = "multi-objective-tpe"
        self.study: Any = None
        self.distributions: Any = None
        self.trial_pool: dict[str, Any] = {}
        self.randseed = self.config.optimize.rand_seed
        self.distributions = create_distributions(self.params)
        self.create_study()

    def create_study(self) -> None:
        """Create the optuna.study object.

        Args:
            None

        Returns:
            None
        """

        sampler = TPESamplerWrapper()
        sampler._rng = LazyRandomStateWrapper(rng=self._rng)
        sampler._random_sampler = RandomSamplerWrapper(rng=self._rng)
        storage_path = str(f"sqlite:///{self.workspace.path}/optuna-{self.study_name}.db")
        storage = optuna.storages.RDBStorage(url=storage_path)
        load_if_exists = self.config.resume is not None

        self.study = optuna.create_study(
            sampler=sampler,
            storage=storage,
            study_name=self.study_name,
            load_if_exists=load_if_exists,
            directions=[item.value for item in self.config.optimize.goal],
        )
