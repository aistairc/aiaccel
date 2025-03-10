# optimize.py

from typing import Any

import argparse
from collections.abc import Callable
import importlib.resources
from pathlib import Path
import pickle as pkl
import subprocess
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from optuna.trial import Trial

from aiaccel.hpo.optuna.suggest_wrapper import Const, Suggest, SuggestFloat, T
from aiaccel.utils import print_config


class HparamsManager:
    """
    Manages hyperparameters for optimization.
    This class allows defining hyperparameters with various types and provides
    a method to suggest hyperparameters for a given trial.
    Attributes:
        params (dict): A dictionary where keys are hyperparameter names and values
                       are callables that take a Trial object and return a hyperparameter value.
    Methods:
        __init__(**params_def: dict[str, int | float | str | list[int | float] | Suggest[T]]) -> None:
            Initializes the HparamsManager with the given hyperparameter definitions.
        suggest_hparams(trial: Trial) -> dict[str, float | int | str | list[float | int | str]]:
            Suggests hyperparameters for the given trial.
    """

    def __init__(self, **params_def: dict[str, int | float | str | list[int | float] | Suggest[T]]) -> None:
        self.params: dict[str, Callable[[Trial], Any]] = {}
        for name, param in params_def.items():
            if callable(param):
                self.params[name] = param
            else:
                if isinstance(param, list):
                    low, high = param
                    self.params[name] = SuggestFloat(name=name, low=low, high=high)
                else:
                    self.params[name] = Const(name=name, value=param)

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

        return {name: param_fn(trial) for name, param_fn in self.params.items()}


def run_job(script_path: str, args: list[str], work_dir: str) -> dict[str, Any]:
    cmd = ["bash", script_path] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=work_dir)
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "returncode": -1}


def main() -> None:
    """
    Main function to execute the hyperparameter optimization process.
    This function parses command-line arguments, loads the configuration,
    sets up the job executor, and runs the optimization trials.

    Command-line arguments:
        - job_filename (Path): The shell script to execute.
        - --config (str, optional): Path to the configuration file.
        - --executor (str, optional): Type of job executor to use ("local", "abci", or "dask").
        - --resume (bool, optional): Flag to resume from the previous study.

    The function performs the following steps:
        1. Parses command-line arguments.
        2. Loads and merges the configuration from the file and command-line arguments.
        3. Sets default storage and study name if not provided in the configuration.
        4. Initializes the job executor based on the specified executor type.
        5. Instantiates the study and parameter suggestion objects.
        6. Submits jobs for hyperparameter optimization trials.
        7. Collects and processes finished jobs, updating the study with results.

    Usage:
        python -m aiaccel.hpo.apps.optimize objective.sh --config config.yaml --executor dask

    Config file (yaml) example:
        ~~~ yaml
        study:
          _target_: optuna.create_study
          direction: minimize

        cluster:
          _target_: dask_jobqueue.PBSCluster
          n_workers: 4
          processes: 1
          queue: "rt_HF"
          memory: "8GB"
          walltime: "02:00:00"

        sampler:
          _target_: optuna.samplers.TPESampler
          seed: 0

        params:
          _convert_: partial
          _target_: aiaccel.apps.optimize.HparamsManager
          x1: [0, 1]
          x2:
            _target_: aiaccel.apps.optimize.SuggestFloat
            name: x2
            low: 0.0
            high: 1.0
            log: false

        n_trials: 30
        n_max_jobs: 4

        group: gaa50000
        ~~~
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("job_filename", type=Path, help="The shell script to execute.")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resumable", action="store_true", default=False)

    args, unk_args = parser.parse_known_args()

    default_config = oc.load(importlib.resources.open_text("aiaccel.hpo.apps.config", "default.yaml"))
    config = oc.merge(default_config, oc.load(args.config) if args.config is not None else {})
    config = oc.merge(config, oc.from_cli(unk_args))

    if (args.resumable or args.resume) and ("storage" not in config.study or args.config is None):
        config = oc.merge(config, oc.load(importlib.resources.open_text("aiaccel.hpo.apps.config", "resumable.yaml")))

    if args.resume:
        config.study.load_if_exists = True

    print_config(config)

    client = instantiate(config.cluster)
    client.cluster.scale(config.n_max_jobs)

    work_dir = Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)

    study = instantiate(config.study)
    params = instantiate(config.params)

    future_to_trial: dict[Any, dict[str, Any]] = {}
    submitted_job_count = 0
    finished_job_count = 0
    result_filename_template = "{job_dir}/{job_name}_result.pkl"

    try:
        while finished_job_count < config.n_trials:
            active_jobs = len([f for f in future_to_trial if not f.done()])
            available_slots = max(0, config.n_max_jobs - active_jobs)
            n_to_submit = min(available_slots, config.n_trials - submitted_job_count)

            for _ in range(n_to_submit):
                trial = study.ask()
                hparams = params.suggest_hparams(trial)
                job_name = f"{args.job_filename.stem}_{trial.number}"
                result_filename = result_filename_template.format(job_dir=work_dir, job_name=job_name)

                cmd_args = [result_filename] + sum([[f"--{k}", f"{v}"] for k, v in hparams.items()], [])

                future = client.submit(run_job, str(args.job_filename), cmd_args, str(work_dir))

                future_to_trial[future] = {"trial": trial, "result_file": result_filename, "job_name": job_name}

                submitted_job_count += 1

            for future in list(future_to_trial.keys()):
                if future.done():
                    trial_info = future_to_trial.pop(future)
                    trial = trial_info["trial"]
                    result_file = trial_info["result_file"]

                    result = future.result()

                    if result["success"]:
                        with open(result_file, "rb") as f:
                            y = pkl.load(f)
                        study._log_completed_trial(study.tell(trial, y))
                    else:
                        study.tell(trial, float("inf"))

                    finished_job_count += 1

            if n_to_submit == 0 and active_jobs == len([f for f in future_to_trial if not f.done()]):
                time.sleep(0.5)

    finally:
        client.close()


if __name__ == "__main__":
    main()
