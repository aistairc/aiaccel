from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.job import AbciJobExecutor

"""
Usage:
    python -m aiaccel.apps.optimize objective.sh x1="[0, 10]" x2="[0, 10]" --config config.yaml
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Process some parameters and execute a shell script.")
    parser.add_argument("filename", type=str, help="The shell script to execute.")
    parser.add_argument("params", nargs="*", help='Additional parameters in the form of key="value"')
    parser.add_argument("--config", nargs="?", default=None)

    args = parser.parse_args()

    params = oc.from_cli(args.params)
    """
    params example:
        params = {'params': {'x': [0, 1], 'y': [0, 1]}}
    """

    config = oc.load(args.config)
    jobs = AbciJobExecutor(Path(args.filename), config.group, n_max_jobs=config.n_max_jobs)
    study = instantiate(config.study)

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    finished_job_count = 0

    while finished_job_count <= config.n_trials:
        n_max_jobs = 1 if config.n_max_jobs == 1 else min(jobs.available_slots(), config.n_trials - finished_job_count)
        for _ in range(n_max_jobs):
            trial = study.ask()
            hparams = {}
            for key, value in params.params.items():
                hparams[key] = trial.suggest_float(key, *value)

            jobs.job_name = str(jobs.job_filename) + f"_{trial.number}"

            job = jobs.submit(
                args=[result_filename_template] + sum([[f"--{k}", f"{v:.5f}"] for k, v in hparams.items()], []),
                tag=trial,
            )

        for job in jobs.collect_finished():
            trial = job.tag

            with open(result_filename_template.format(job=job), "rb") as f:
                y = pkl.load(f)

            study.tell(trial, y)

            finished_job_count += 1


if __name__ == "__main__":
    main()
