#! /usr/bin/env python3

from typing import Any

from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

default_config_yaml = """\
script_prologue: |
    echo Hostname: $(hostname)

    export NVIDIA_VISIBLE_DEVICES=all

cpu:
    job: "{command}"

cpu-array:
    n_tasks_per_proc: 128
    n_procs: 24
    job: "{command}"

gpu:
    job: "{command}"

gpu-array:
    n_tasks_per_proc: 128
    n_procs: 8
    job: "CUDA_VISIBLE_DEVICES=$(( LOCAL_PROC_INDEX % {args.n_procs_per_job} )) {command}"

mpi:
    job: |
        mpirun -np {args.n_procs} \\
            {command}

train:
    job: |
        mpirun -np {args.n_gpus} \\
            -x MAIN_ADDR=$(hostname -i) \\
            -x MAIN_PORT=3000 \\
            -x COLUMNS=120 \\
            -x PYTHONUNBUFFERED=true \\
            {command}
"""  # noqa: E501


def dispatch_job(mode: str, args: Namespace, config: Any) -> None:
    # Prepare the job script and arguments
    job = config[mode]["job"].format(command=shlex.join(args.command), args=args)

    if mode in ["cpu-array", "gpu-array"]:
        job = f"TASK_STEPSIZE={args.n_tasks_per_proc} {job}"
        log_filename = f"{args.log_filename.with_suffix('')}.${{TASK_INDEX}}.log"
    else:
        job = f"{job} > {args.log_filename} 2>&1"
        log_filename = args.log_filename

    job_script = f"""\
#! /bin/bash

set -eE -o pipefail
trap 'exit $?' ERR EXIT  # at error and exit
trap 'echo 143' TERM  # at termination (by job scheduler)


{config["script_prologue"]}

{job} 2>&1 | tee {log_filename}
"""

    # Create the job script file, remove old status files, and run the job
    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    job_filename: Path = args.log_filename.with_suffix(".sh")
    with open(job_filename, "w") as f:
        f.write(job_script)

    if mode in ["cpu-array", "gpu-array"]:
        worker = partial(subprocess.run, shell=True, check=True)
        with Pool(processes=args.n_procs) as pool:
            cmd_list = []
            for ii in range(0, args.n_tasks, args.n_tasks_per_proc):
                cmd_list.append(f"TASK_INDEX={ii + 1} bash {job_filename}")

            for _ in pool.imap_unordered(worker, cmd_list):
                pass
    else:
        subprocess.run(f"bash {job_filename}", shell=True, check=True)


def main() -> None:
    # Load configuration (from the default YAML string)
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path)
    args, _ = parser.parse_known_args()

    if args.config is not None:
        with open(args.config) as f:
            config_yaml = f.read()
    else:
        config_yaml = default_config_yaml

    if args.print_config:
        print(config_yaml)
        sys.exit(0)

    config = yaml.safe_load(config_yaml)

    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path)
    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--walltime", type=str)  # defined for compatibility
    parent_parser.add_argument("log_filename", type=Path)
    parent_parser.add_argument("command", nargs="+")

    sub_parser = sub_parsers.add_parser("cpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["cpu-array"]["n_tasks_per_proc"])
    sub_parser.add_argument("--n_procs", type=int, default=config["cpu-array"]["n_procs"])

    sub_parser = sub_parsers.add_parser("gpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["gpu-array"]["n_tasks_per_proc"])
    sub_parser.add_argument("--n_procs", type=int, default=config["gpu-array"]["n_procs"])

    sub_parser = sub_parsers.add_parser("mpi", parents=[parent_parser])
    sub_parser.add_argument("--n_procs", type=int, required=True)
    sub_parser.add_argument("--n_nodes", type=int)  # defined for compatibility

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    args = parser.parse_args()
    mode = args.mode + "-array" if getattr(args, "n_tasks", None) is not None else args.mode

    dispatch_job(mode, args, config)


if __name__ == "__main__":
    main()
