""" Example Optuna ask-and-tell loop
"""

import optuna


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    for _ in range(100):
        trial = study.ask()
        y = objective(trial)
        study.tell(trial, y)
