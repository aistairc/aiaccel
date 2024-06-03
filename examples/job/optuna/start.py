from __future__ import annotations

import argparse
import ast
import pickle as pkl
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import optuna
from omegaconf.dictconfig import DictConfig

from aiaccel.config import load_config
from aiaccel.job import AbciJob, AbciJobExecutor

"""
Usage:
    python start.py objective.sh x1="[0, 10]" x2="[0, 10]" --config config.yaml
"""

result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"


def import_and_getattr(search_algorithm: str) -> Any | None:
    """
    Import a module and get an attribute from it.

    Args:
        search_algorithm (str): The module and attribute to import.

    Returns:
        Any | None: The attribute from the module.
    """
    module_name, attr_name = search_algorithm.rsplit(".", 1)
    module = import_module(module_name)

    return getattr(module, attr_name)


def params_to_dict(arg_params: list[str]) -> dict[str, Any]:
    """
    Convert a list of parameters to a dictionary.

    Args:
        arg_params (list[Any]): The list of parameters.
        e.g. ["x1=[0, 10]", "x2=[0, 10]"]

    Returns:
        dict[str, Any]: The dictionary of parameters.
        e.g. {"x1": [0, 10], "x2": [0, 10]}
    """
    params_dict = {}
    for param in arg_params:
        key, value = param.split("=", 1)
        try:
            # Convert string to list
            params_dict[key] = ast.literal_eval(value)
        except Exception as e:
            print(f"Error parsing parameter {key}: {e}")
            sys.exit(1)

    return params_dict


def execute_serially(filename: Path, config: DictConfig, params: dict[str, Any]) -> None:
    """
    Execute the shell script serially.

    Args:
        filename (Path): The path to the shell script.
        config (DictConfig): The configuration for the job.
        params (dict[str, Any]): The parameters to optimize.

    Raises:
        ValueError: If the sampler is not found.
    """

    sampler = import_and_getattr(config.sampler)
    if sampler is None:
        raise ValueError("Sampler not found.")

    study = optuna.create_study(sampler=sampler(), direction=config.direction)

    for _ in range(config.n_trials):
        trial = study.ask()

        hparams = {}
        for key, value in params.items():
            hparams[key] = trial.suggest_float(key, *value)

        job = AbciJob(
            filename,
            config.group,
            args=[result_filename_template] + sum([[f"--{k}", f"{v:.5f}"] for k, v in hparams.items()], []),
        )

        job.submit()
        job.wait()

        with open(result_filename_template.format(job=job), "rb") as f:
            y = pkl.load(f)

        study.tell(trial, y)


def execute_parallelly(filename: Path, config: DictConfig, params: dict[str, Any]) -> None:
    """
    Execute the shell script in parallel.

    Args:
        filename (Path): The path to the shell script.
        config (DictConfig): The configuration for the job.
        params (dict[str, Any]): The parameters to optimize.

    Raises:
        ValueError: If the sampler is not found.
    """

    jobs = AbciJobExecutor(filename, config.group, n_max_jobs=config.n_max_jobs)

    sampler = import_and_getattr(config.sampler)
    if sampler is None:
        raise ValueError("Sampler not found.")

    study = optuna.create_study(sampler=sampler(), direction=config.direction)

    finished_job_count = 0

    while finished_job_count <= config.n_trials:
        for _ in range(jobs.available_slots()):
            trial = study.ask()
            hparams = {}
            for key, value in params.items():
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
    parser = argparse.ArgumentParser(description="Process some parameters and execute a shell script.")
    parser.add_argument("filename", type=str, help="The shell script to execute.")
    parser.add_argument("params", nargs="*", help='Additional parameters in the form of key="value"')
    parser.add_argument("--config", nargs="?", default=None)

    args = parser.parse_args()

    params = params_to_dict(args.params)
    if len(params) == 0:
        print("No parameters provided. Usage: python start.py <filename> <key1=value1> <key2=value2> ...")
        sys.exit(1)

    config = load_config(args.config)

    if config.n_max_jobs == 1:
        execute_serially(Path(args.filename), config, params)
    else:
        execute_parallelly(Path(args.filename), config, params)
