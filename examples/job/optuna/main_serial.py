<<<<<<< HEAD
import pickle as pkl
from pathlib import Path
=======
from pathlib import Path
import pickle as pkl
>>>>>>> 0caccda (update v2 torch   (#383))

import optuna

from aiaccel.job import AbciJob


def main() -> None:
    n_trials = 50

    job_filename = Path("objective.sh")
    job_group = "gaa50000"

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    study = optuna.create_study(direction="minimize")

    for _ in range(n_trials):
        trial = study.ask()
        hparams = {
            "x1": trial.suggest_float("x1", 0, 10),
            "x2": trial.suggest_float("x2", 0, 10),
        }

        job = AbciJob(
            job_filename,
            job_group,
            args=[result_filename_template] + sum([[f"--{k}", f"{v:.5f}"] for k, v in hparams.items()], []),
        )

        job.submit()
        job.wait()

        with open(result_filename_template.format(job=job), "rb") as f:
            y = pkl.load(f)

        study.tell(trial, y)


if __name__ == "__main__":
    main()
