import pickle as pkl
from pathlib import Path

import optuna

from aiaccel.job import AbciJobExecutor


def main() -> None:
    n_trials = 50

    job_filename = Path("objective.sh")
    job_group = "gaa50000"

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    jobs = AbciJobExecutor(job_filename, job_group)

    study = optuna.create_study(direction="minimize")

    for _ in range(n_trials):
        for _ in range(jobs.available_slots()):
            trial = study.ask()
            hparams = {
                "x1": trial.suggest_float("x1", 0, 10),
                "x2": trial.suggest_float("x2", 0, 10),
            }

            job = jobs.submit(
                args=[result_filename_template] + sum([[f"--{k}", f"{v:.5f}"] for k, v in hparams.items()], []),
                tag=trial,
            )

        for job in jobs.collect_finished():
            trial = job.tag

            with open(result_filename_template.format(job=job), "rb") as f:
                y = pkl.load(f)

            study.tell(trial, y)


if __name__ == "__main__":
    main()
