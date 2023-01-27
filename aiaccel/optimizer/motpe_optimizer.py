import optuna
from aiaccel.optimizer.tpe_optimizer import TpeOptimizer, TPESamplerWrapper


class MOTpeOptimizer(TpeOptimizer):
    def __init__(self, options: dict) -> None:
        """Initial method of TpeOptimizer.

        Args:
            options (dict): A file name of a configuration.
        """
        super().__init__(options)
        self.study_name = "multi-objective-tpe"

    def create_study(self) -> None:
        """Create the optuna.study object and store it.

        Returns:
            None
        """
        if self.study is None:
            self.study = optuna.create_study(
                sampler=TPESamplerWrapper(seed=self.randseed),
                study_name=self.study_name,
                directions=self.config.goal.get()
            )
