from typing import Any

from collections.abc import Callable

from optuna.trial import Trial

from aiaccel.hpo.optuna.suggest_wrapper import Const, Float, Hparam, T


class HparamsManager:
    """
    Manages hyperparameters for optimization.
    This class allows defining hyperparameters with various types and provides
    a method to suggest hyperparameters for a given trial.
    Attributes:
        params (dict): A dictionary where keys are hyperparameter names and values
                       are callables that take a Trial object and return a hyperparameter value.
    Methods:
        __init__(**params_def: dict[str, int | float | str | list[int | float] | Hparam[T]]) -> None:
            Initializes the HparamsManager with the given hyperparameter definitions.
        suggest_hparams(trial: Trial) -> dict[str, float | int | str | list[float | int | str]]:
            Suggests hyperparameters for the given trial.
    """

    def __init__(self, **params_def: dict[str, int | float | str | list[int | float] | Hparam[T]]) -> None:
        self.params: list[Callable[[Trial], Any]] = []
        for name, param in params_def.items():
            if callable(param):
                self.params.append(param)
            else:
                if isinstance(param, list):
                    low, high = param
                    self.params.append(Float(name=name, low=low, high=high))
                else:
                    self.params.append(Const(name=name, value=param))

    def suggest_hparams(self, trial: Trial) -> dict[str, float | int | str | list[float | int | str]]:
        """
        Suggests hyperparameters for a given trial.
        This method generates a dictionary of hyperparameters by applying the
        parameter functions stored in `self.params` to the provided trial.
        Args:
            trial (Trial): An Optuna trial object used to suggest hyperparameters.
        Returns:
            dict[str, float | int | str | list[float | int | str]]: A dictionary
            where keys are parameter names and values are the suggested
            hyperparameters, which can be of type float, int, str, or a list of
            these types.
        """

        params_dict = {}
        for param_fn in self.params:
            if not hasattr(param_fn, "name"):
                raise AttributeError(f"{param_fn} has no attribute 'name'")
            params_dict[param_fn.name] = param_fn(trial)
        return params_dict
