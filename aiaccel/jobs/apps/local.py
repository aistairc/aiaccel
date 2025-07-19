#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

default_config_yaml = """\
script_template: |
    #! /bin/bash

    set -e

    trap 'echo $? > {status_filename}' EXIT

    echo Hostname: $(hostname)

    export NVIDIA_VISIBLE_DEVICES=all

    export SINGULARITYENV_COLUMNS=120
    export SINGULARITYENV_PYTHONUNBUFFERED=true

    {job}

cpu:
  n_tasks_per_proc: 128
  n_procs_per_job: 48
  job_template: |
    {command}

gpu:
  n_tasks_per_proc: 128
  n_procs_per_job: 8
  job_template: |
    {command}

mpi:
  n_nodes: 1
  job_template: |
    mpirun -np {args.n_procs} \\
        {command}

train:
  job_template: |
    mpirun -np {args.n_gpus} \\
        {command}
"""


def main() -> None:
    # Pre-parser to load configuration in advance.
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path)
    args, _ = parser.parse_known_args()

    if args.print_config:
        print(default_config_yaml)
        sys.exit(0)

    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = yaml.safe_load(default_config_yaml)

    # Main parser
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--print_config", action="store_true")
    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--walltime", type=str)  # dummy
    parent_parser.add_argument("log_filename", type=Path)
    parent_parser.add_argument("command", nargs="+")

    sub_parser = sub_parsers.add_parser("cpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["cpu"]["n_tasks_per_proc"])
    sub_parser.add_argument("--n_procs_per_job", type=int, default=config["cpu"]["n_procs_per_job"])

    sub_parser = sub_parsers.add_parser("gpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["gpu"]["n_tasks_per_proc"])
    sub_parser.add_argument("--n_procs_per_job", type=int, default=config["gpu"]["n_procs_per_job"])

    sub_parser = sub_parsers.add_parser("mpi", parents=[parent_parser])
    sub_parser.add_argument("--n_procs", type=int, required=True)
    sub_parser.add_argument("--n_nodes", type=int, default=config["mpi"]["n_nodes"])

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    args = parser.parse_args()

    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    job = config[args.mode]["job_template"].format(args=args, command=shlex.join(args.command))

    job_filename: Path = args.log_filename.with_suffix(".sh")
    status_filename: Path = args.log_filename.with_suffix(".out")
    with open(job_filename, "w") as f:
        f.write(config["script_template"].format(status_filename=status_filename, job=job))

    try:
        with open(args.log_filename, "wb", buffering=0) as f:
            subprocess.run(["bash", str(job_filename)], check=True, stdout=f, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Job failed with exit code {e.returncode}. Check {args.log_filename} for details.", file=sys.stderr)

        exit(e.returncode)


if __name__ == "__main__":
    main()
