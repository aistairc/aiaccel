import optuna

from aiaccel.hpo.optuna.suggest_wrapper import (
    Const,
    SuggestCategorical,
    SuggestDiscreteUniform,
    SuggestFloat,
    SuggestInt,
    SuggestUniform,
)


def test_const() -> None:
    const = Const(name="x1", value=0.5)
    assert const(None) == 0.5


def test_suggest_float() -> None:
    suggest_float = SuggestFloat(name="x2", low=0.0, high=1.0, step=None, log=False)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_float(trial), float)


def test_suggest_int() -> None:
    suggest_int = SuggestInt(name="x3", low=0, high=10, step=1, log=False)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_int(trial), int)


def test_suggest_categorical() -> None:
    suggest_categorical = SuggestCategorical(name="x4", choices=[0, 1, 2])
    trial = optuna.create_study().ask()

    assert suggest_categorical(trial) in [0, 1, 2]


def test_suggest_discrete_uniform() -> None:
    suggest_discrete_uniform = SuggestDiscreteUniform(name="x5", low=0.0, high=1.0, q=0.1)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_discrete_uniform(trial), float)


def test_suggest_log_uniform() -> None:
    suggest_log_uniform = SuggestDiscreteUniform(name="x6", low=0.0, high=1.0, q=0.1)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_log_uniform(trial), float)


def test_suggest_uniform() -> None:
    suggest_uniform = SuggestUniform(name="x7", low=0.0, high=1.0)
    trial = optuna.create_study().ask()

    assert isinstance(suggest_uniform(trial), float)
