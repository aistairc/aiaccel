#! /usr/bin/env python

from typing import Any

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

default_config_yaml = """\
walltime: 1:0:0

script_prologue: |
    export NVIDIA_VISIBLE_DEVICES=all

qsub: "qsub -P $JOB_GROUP -l walltime={args.walltime} -o {log_filename} -j oe -k oed -v USE_SSH=1"

cpu:
    qsub_args: "-q rt_HF -l select=1"
    job: "{command}"

cpu-array:
    n_tasks_per_proc: 128
    n_procs_per_job: 48
    qsub_args: "-q rt_HF -l select=1 -J 1-{args.n_tasks}:{args.n_tasks_per_proc}"
    job: "{command}"

gpu:
    qsub_args: "-q rt_HF -l select=1"
    job: "{command}"

gpu-array:
    n_tasks_per_proc: 128
    n_procs_per_job: 48
    qsub_args: "-q rt_HF -l select=1 -J 1-{args.n_tasks}:{args.n_tasks_per_proc}"
    job: "CUDA_VISIBLE_DEVICES=$(( LOCAL_PROC_INDEX % 8 )) {command}"

mpi:
    n_nodes: 1
    qsub_args: "-q rt_HF -l select={args.n_nodes}:mpiprocs=$(( {args.n_procs} / args.n_nodes )):ompthreads=$(( {args.n_procs} * 96 / args.n_nodes ))"
    job: |
        source /etc/profile.d/modules.sh
        module load hpcx

        mpirun -np {args.n_procs} -bind-to none -map-by slot \\
            -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
            {command}

train:
    qsub_args: "-q $( ((N==1)) && printf rt_HG || printf rt_HF ) -l select=$(( ({args.n_gpus} + 7) / 8 )):mpiprocs=8:ompthreads=12"
    job: |
        source /etc/profile.d/modules.sh
        module load hpcx

        mpirun -np {args.n_gpus} -bind-to none -map-by slot \\
            -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
            -x MAIN_ADDR=$(hostname -i) \\
            -x MAIN_PORT=3000 \\
            -x COLUMNS=120 \\
            -x PYTHONUNBUFFERED=true \\
            {command}
"""  # noqa: E501


def load_config() -> Any:
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

    return yaml.safe_load(config_yaml)


def parse_args(config: Any) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path)
    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--local", action="store_true")
    parent_parser.add_argument("--walltime", type=str, default=config["walltime"])
    parent_parser.add_argument("log_filename", type=Path)
    parent_parser.add_argument("command", nargs="+")

    sub_parser = sub_parsers.add_parser("cpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["cpu-array"]["n_tasks_per_proc"])
    sub_parser.add_argument("--n_procs_per_job", type=int, default=config["cpu-array"]["n_procs_per_job"])

    sub_parser = sub_parsers.add_parser("gpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["gpu-array"]["n_tasks_per_proc"])
    sub_parser.add_argument("--n_procs_per_job", type=int, default=config["gpu-array"]["n_procs_per_job"])

    sub_parser = sub_parsers.add_parser("mpi", parents=[parent_parser])
    sub_parser.add_argument("--n_procs", type=int, required=True)
    sub_parser.add_argument("--n_nodes", type=int, default=config["mpi"]["n_nodes"])

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    return parser.parse_args()


def prepare_array_job(job: str, args: Namespace) -> tuple[Path, Path, str, list[Path]]:
    status_filename: Path = args.log_filename.with_suffix(".${PBS_ARRAY_INDEX}.out")
    log_filename = args.log_filename.with_suffix(".${PBS_ARRAY_INDEX}.proc-${LOCAL_PROC_INDEX}.log")

    job = f"""\
for LOCAL_PROC_INDEX in {{1..{args.n_procs_per_job}}}; do
    TASK_INDEX=$(( PBS_ARRAY_INDEX + {args.n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={args.n_tasks_per_proc} \\
        {job} > {log_filename} 2>&1 &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""

    status_filename_list = []
    for array_idx in range(0, args.n_tasks, args.n_tasks_per_proc):
        status_filename_list.append(args.log_filename.with_suffix(f".{array_idx + 1}.out"))

    return status_filename, log_filename, job, status_filename_list


def prepare_job(job: str, args: Namespace) -> tuple[Path, Path, str, list[Path]]:
    log_filename = args.log_filename
    status_filename = args.log_filename.with_suffix(".out")

    status_filename_list = [status_filename]

    return status_filename, log_filename, job, status_filename_list


def main() -> None:
    config = load_config()
    args = parse_args(config)

    mode = args.mode + "-array" if getattr(args, "n_tasks", None) is not None else args.mode

    job = config[mode]["job"].format(command=shlex.join(args.command), args=args)

    if mode.endswith("-array"):
        status_filename, log_filename, job, status_filename_list = prepare_array_job(job, args)
    else:
        status_filename, log_filename, job, status_filename_list = prepare_job(job, args)

    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    job_filename: Path = args.log_filename.with_suffix(".sh")
    with open(job_filename, "w") as f:
        f.write(f"""\
#! /bin/bash

#PBS -j oe
#PBS -k oed

set -eE
trap 'echo $? > {status_filename}' ERR EXIT  # at error and exit
trap 'echo 143 > {status_filename}' TERM  # at termination (by job scheduler)

if [ -n "$PBS_O_WORKDIR" ] && [ "$PBS_ENVIRONMENT" != "PBS_INTERACTIVE" ]; then
    cd $PBS_O_WORKDIR
fi

echo Job ID: $PBS_JOBID
echo Hostname: $(hostname)

{config["script_prologue"]}

{job}
""")

    for status_filename in status_filename_list:
        status_filename.unlink(missing_ok=True)

    qsub_command = config["qsub"].format(log_filename=log_filename, args=args)
    qsub_command += " " + config[mode]["qsub_args"].format(args=args)

    if not args.local:
        subprocess.run(f"echo {log_filename} {job_filename}", shell=True, check=True)

        for status_filename in status_filename_list:
            while not status_filename.exists():
                time.sleep(1.0)

            status = status_filename.read_text()
            if int(status) != 0:
                raise RuntimeError(f"Job failed with {status} exit code.")
            status_filename.unlink()
    else:
        subprocess.run(f"bash {job_filename}", shell=True, check=True)


if __name__ == "__main__":
    main()
