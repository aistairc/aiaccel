from typing import Any

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from importlib import resources
import json
from pathlib import Path
import shlex
import subprocess

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from optuna.trial import Trial

from aiaccel.config import load_config, print_config, resolve_inherit


def main() -> None:
    """
    Main function to execute the hyperparameter optimization process using a Dask cluster.
    This function parses command-line arguments, loads the configuration,
    sets up the Dask client, and runs the optimization trials in a distributed manner.

    Command-line arguments:
        - --config (str, optional): Path to the configuration file.
        - --resume (bool, optional): Flag to resume from the previous study.
        - --resumable (bool, optional): Flag to make the study resumable by setting appropriate storage.

    Usage:
        - Start a new study:
            python -m aiaccel.hpo.apps.optimize --config config.yaml
        - Resume from the previous study:
            python -m aiaccel.hpo.apps.optimize --config config.yaml --resume
        - Make the study resumable:
            python -m aiaccel.hpo.apps.optimize --config config.yaml --resumable

    Config file (yaml) example:
        ~~~ yaml
        study:
          _target_: optuna.create_study
          direction: minimize
          storage:
            _target_: optuna.storages.InMemoryStorage
          study_name: aiaccel_study
          load_if_exists: false

        cluster:
          _target_: distributed.Client
          n_workers: 4
          threads_per_worker: 1

        sampler:
          _target_: optuna.samplers.TPESampler
          seed: 0

        params:
          _convert_: partial
          _target_: aiaccel.hpo.optuna.hparams_manager.HparamsManager
          x1: [0, 1]
          x2:
            _target_: aiaccel.apps.optimize.Float
            name: x2
            low: 0.0
            high: 1.0
            log: false

        n_trials: 30
        n_max_jobs: 4
        ~~~
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+")

    parser.add_argument("--config", help="Configuration file path", default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resumable", action="store_true", default=False)

    args, unk_args = parser.parse_known_args()

    with resources.as_file(resources.files("aiaccel.hpo.apps.config") / "default.yaml") as path:
        default_config = oc.load(path)
    config = oc.merge(default_config, load_config(args.config) if args.config is not None else {})
    config = oc.merge(config, oc.from_cli(unk_args))

    if (args.resumable or args.resume) and ("storage" not in config.study or args.config is None):
        with resources.as_file(resources.files("aiaccel.hpo.apps.config") / "resumable.yaml") as path:
            config = oc.merge(config, path)

    if args.resume:
        config.study.load_if_exists = True

    print_config(config)

    config = resolve_inherit(config)

    work_dir = Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)

    study = instantiate(config.study)
    params = instantiate(config.params)

    futures: dict[Any, tuple[Trial, str]] = {}
    submitted_job_count = 0
    finished_job_count = 0

    with ThreadPoolExecutor(config.n_max_jobs) as pool:
        while finished_job_count < config.n_trials:
            active_jobs = len(futures.keys())
            available_slots = max(0, config.n_max_jobs - active_jobs)

            # Submit job in ThreadPoolExecutor
            for _ in range(min(available_slots, config.n_trials - submitted_job_count)):
                trial = study.ask()
                out_filename = config.out_filename_template.format(**vars())

                future = pool.submit(
                    subprocess.run,
                    shlex.join(args.command).format(
                        job_name=config.job_name_template.format(**vars()),
                        out_filename=out_filename,
                        **params.suggest_hparams(trial),
                    ),
                    shell=True,
                )

                futures[future] = trial, out_filename
                submitted_job_count += 1

            # Get result from out_filename and tell
            done_features, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done_features:
                trial, out_filename = futures.pop(future)

                with open(out_filename) as f:
                    y = json.load(f)

                study._log_completed_trial(study.tell(trial, y))
                finished_job_count += 1


if __name__ == "__main__":
    main()
