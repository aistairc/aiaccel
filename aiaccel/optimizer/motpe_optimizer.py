import optuna

from aiaccel.optimizer.tpe_optimizer import TpeOptimizer, TPESamplerWrapper


class MOTpeOptimizer(TpeOptimizer):
    """An optimizer class based on multi-objective optuna.samplers.TPESampler.

    Args:
        options (dict): A dictionary containing command line options.
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        """Initial method of MOTpeOptimizer.

        Args:
            options (dict): A file name of a configuration.
        """
        super().__init__(options)
        self.study_name = "multi-objective-tpe"

    def create_study(self) -> None:
        """Create the optuna.study object.

        Returns:
            None
        """

        sampler = TPESamplerWrapper()
        sampler._rng = self._rng
        sampler._random_sampler._rng = self._rng
        storage_path = str(f"sqlite:///{self.ws}/optuna-{self.study_name}.db")
        storage = optuna.storages.RDBStorage(url=storage_path)
        load_if_exists = self.options["resume"] is not None

        self.study = optuna.create_study(
            sampler=sampler,
            storage=storage,
            study_name=self.study_name,
            load_if_exists=load_if_exists,
            directions=self.config.goal.get(),
        )
