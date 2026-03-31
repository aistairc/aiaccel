# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import cast

from argparse import ArgumentParser, _SubParsersAction
from importlib import resources
import os
from pathlib import Path
import subprocess
import time

from omegaconf import DictConfig

from aiaccel.config import prepare_config, print_config, setup_omegaconf

setup_omegaconf()


def prepare_argument_parser(
    default_config_name: str,
) -> tuple[DictConfig, ArgumentParser, _SubParsersAction]:  # type: ignore
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args()

    args.config = Path(
        args.config
        or os.environ.get("AIACCEL_JOB_CONFIG")
        or (Path(str(resources.files(__package__) / "config")) / default_config_name)
    )  # type: ignore

    config = cast(DictConfig, prepare_config(args.config))

    if args.print_config:
        print_config(config)

    parser = ArgumentParser()
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path)
    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--walltime", type=str, default=config.walltime)
    parent_parser.add_argument("log_filename", type=Path)
    parent_parser.add_argument("command", nargs="+")

    sub_parser = sub_parsers.add_parser("cpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["cpu-array"].n_tasks_per_proc)
    sub_parser.add_argument("--n_procs", type=int, default=config["cpu-array"].n_procs)

    sub_parser = sub_parsers.add_parser("gpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["gpu-array"].n_tasks_per_proc)
    sub_parser.add_argument("--n_procs", type=int, default=config["gpu-array"].n_procs)

    sub_parser = sub_parsers.add_parser("mpi", parents=[parent_parser])
    sub_parser.add_argument("--n_procs", type=int, required=True)
    sub_parser.add_argument("--n_nodes", type=int, default=config["mpi"].n_nodes)

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    return config, parser, sub_parsers


def _is_skip_job_submission(job_filename: Path, job_script: str, status_filename_list: list[Path]) -> bool:
    has_same_job_script = job_filename.exists() and job_filename.read_text() == job_script
    has_success_status_files = all(
        status_filename.exists() and status_filename.read_text().strip() == "0"
        for status_filename in status_filename_list
    )
    return has_same_job_script and has_success_status_files


def _submit_job_and_wait(
    log_filename: Path,
    job_script: str,
    submit_command: str,
    submit_args: str,
    status_filename_list: list[Path],
    use_scandir: bool,
) -> None:
    job_filename = log_filename.with_suffix(".sh")
    if _is_skip_job_submission(job_filename, job_script, status_filename_list):
        print(
            "A successfully completed .out file exists"
            f"({[str(status_filename) for status_filename in status_filename_list]}), "
            "so the job will not be submitted."
        )
        for status_filename in status_filename_list:
            status_filename.unlink(missing_ok=True)
        return

    log_filename.parent.mkdir(exist_ok=True, parents=True)

    with open(job_filename, "w") as f:
        f.write(job_script)

    for status_filename in status_filename_list:
        status_filename.unlink(missing_ok=True)

    subprocess.run(f"{submit_command} {submit_args} {job_filename}", shell=True, check=True)

    for status_filename in status_filename_list:
        while not status_filename.exists():
            time.sleep(1.0)

            if use_scandir:  # Reflesh the file system if needed
                os.scandir(status_filename.parent)

        status = int(status_filename.read_text())
        if status != 0:
            raise RuntimeError(f"Job failed with {status} exit code.")
        status_filename.unlink()
