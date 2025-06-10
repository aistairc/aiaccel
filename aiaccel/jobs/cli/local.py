#! /usr/bin/env python

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
import subprocess

from omegaconf import DictConfig, ListConfig

from aiaccel.config.config import load_config


def prepare_mpi_job(args: Namespace, command: str) -> str:
    mpi_args = []
    if isinstance(args.mpi_args_path, str | Path):
        mpi_args_dest = load_config(args.mpi_args_path)
        if isinstance(mpi_args_dest, DictConfig):
            for key, value in mpi_args_dest.items():
                if isinstance(value, ListConfig):
                    for v in value:
                        mpi_args.append(f"{key!r} {v} \\")
                else:
                    mpi_args.append(f"{key!r} {value} \\")
        else:
            mpi_args.append(str(mpi_args_dest))

    joined_mpi_args = "\n".join(mpi_args)
    job = f"""
    mpirun {joined_mpi_args}
    {command}
    """

    return job


def main() -> None:
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--local", action="store_true")
    parent_parser.add_argument("--command_prefix", type=str)
    parent_parser.add_argument("--walltime", type=str, default="0:40:0")
    parent_parser.add_argument("--log_filename", type=Path)
    parent_parser.add_argument("--command", nargs="+")

    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    sub_parser = sub_parsers.add_parser("cpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=2048)
    sub_parser.add_argument("--n_procs_per_job", type=int, default=32)

    sub_parser = sub_parsers.add_parser("gpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=1024)
    sub_parser.add_argument("--n_procs_per_job", type=int, default=8)

    sub_parser = sub_parsers.add_parser("mpi", parents=[parent_parser])
    sub_parser.add_argument("--n_procs", type=int, required=True)
    sub_parser.add_argument("--n_nodes", type=int, default=1)
    sub_parser.add_argument("--mpi_args_path", default=None)

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)
    sub_parser.add_argument("--mpi_args_path", default=None)

    args = parser.parse_args()

    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    command = f"{args.command_prefix} {shlex.join(args.command)}"

    if args.mode in ["cpu", "gpu"]:
        job = command
    elif args.mode == "mpi" or args.mode == "train":
        job = prepare_mpi_job(args, command)
    else:
        raise ValueError()

    job_filename: Path = args.log_filename.with_suffix(".sh")
    status_filename: Path = args.log_filename.with_suffix(".out")
    with open(job_filename, "w") as f:
        f.write(
            f"""\
#! /bin/bash

set -e

trap 'echo $? > {status_filename}' EXIT

echo Hostname: $(hostname)

export NVIDIA_VISIBLE_DEVICES=all

export SINGULARITYENV_COLUMNS=120
export SINGULARITYENV_PYTHONUNBUFFERED=true

{job}
"""
        )

    subprocess.run(["bash", str(job_filename)], check=True)


if __name__ == "__main__":
    main()
