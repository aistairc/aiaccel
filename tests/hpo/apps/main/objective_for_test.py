from pathlib import Path
import pickle as pkl

import optuna
from optuna.trial import Trial


def main(x1: float, x2: float) -> float:
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    return y


def objective(trial: Trial) -> float:
    x1 = trial.suggest_float("x1", 0.0, 1.0)
    x2 = trial.suggest_float("x2", 0.0, 1.0)
    return main(x1, x2)


if __name__ == "__main__":
    # ==========================================================
    # normal
    db_path = Path("./test_normal.db")
    study_name = "test_study_normal"
    url = f"sqlite:///{db_path}"

    if db_path.exists():
        db_path.unlink()

    storage = optuna.storages.RDBStorage(url=url)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0), storage=storage, study_name=study_name
    )

    for _ in range(30):
        trial = study.ask()
        y = main(x1=trial.suggest_float("x1", 0.0, 1.0), x2=trial.suggest_float("x2", 0.0, 1.0))
        study._log_completed_trial(study.tell(trial, y))

    with open("./test_notmal.pkl", "wb") as f:
        pkl.dump(study, f)

    # ==========================================================
    # resume
    db_path = Path("./test_resume.db")
    study_name = "test_study_resume"
    url = f"sqlite:///{db_path}"

    if db_path.exists():
        db_path.unlink()

    storage = optuna.storages.RDBStorage(url=url)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0), storage=storage, study_name=study_name
    )

    for _ in range(15):
        trial = study.ask()
        y = main(x1=trial.suggest_float("x1", 0.0, 1.0), x2=trial.suggest_float("x2", 0.0, 1.0))
        study._log_completed_trial(study.tell(trial, y))

    # resume
    print("resume after 14 trials")
    study = optuna.load_study(study_name=study_name, sampler=optuna.samplers.TPESampler(seed=0), storage=url)

    for _ in range(15):
        trial = study.ask()
        y = main(x1=trial.suggest_float("x1", 0.0, 1.0), x2=trial.suggest_float("x2", 0.0, 1.0))
        study._log_completed_trial(study.tell(trial, y))

    with open("./test_resume.pkl", "wb") as f:
        pkl.dump(study, f)

    # ==========================================================
    # compare
    with open("./test_resume.pkl", "rb") as f:
        study_resume = pkl.load(f)
    with open("./test_notmal.pkl", "rb") as f:
        study_normal = pkl.load(f)

    try:
        assert study_resume.best_value == study_normal.best_value, (
            f"resume: {study_resume.best_value}, normal: {study_normal.best_value}"
        )
    except AssertionError as e:
        print(e)
