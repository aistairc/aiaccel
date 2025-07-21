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
walltime: "1:0:0"

script_prologue: |
    echo Job ID: $JOB_ID
    echo Hostname: $(hostname)

    export NVIDIA_VISIBLE_DEVICES=all

qsub: "qsub -g $JOB_GROUP -l h_rt={args.walltime}"

cpu:
    qsub_args: "-l cpu_40=1"
    job: "{command}"

cpu-array:
    n_tasks_per_proc: 128
    n_procs: 20
    qsub_args: "-l cpu_40=1 -t 1-{args.n_tasks}:$(( {args.n_tasks_per_proc} * {args.n_procs} ))"
    job: "{command}"

gpu:
    qsub_args: "-l gpu_1=1"
    job: "{command}"

gpu-array:
    n_tasks_per_proc: 128
    n_procs: 1
    qsub_args: "-l gpu_1=1 -t 1-{args.n_tasks}:$(( {args.n_tasks_per_proc} * {args.n_procs} ))"
    job: "{command}"

mpi:
    n_nodes: 1
    qsub_args: "-l cpu_40={args.n_nodes}"
    job: |
        source /etc/profile.d/modules.sh
        module load openmpi

        mpirun -np {args.n_procs} --npernode $(( {args.n_procs} / {args.n_nodes} )) \\
            -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
            {command}

train:
    qsub_args: "-l $( (({args.n_gpus}==1)) && printf node_q || printf node_f )=$(( ({args.n_gpus} + 3) / 4 ))"
    job: |
        source /etc/profile.d/modules.sh
        module load openmpi

        n_gpus=$(nvidia-smi -L | wc -l)

        mpirun -np {args.n_gpus} -map-by ppr:$n_gpus:node:PE=48 \\
            -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
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
        job = f"""\
for LOCAL_PROC_INDEX in {{1..{args.n_procs}}}; do
    TASK_INDEX=$(( SGE_TASK_ID + {args.n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={args.n_tasks_per_proc} \\
        {job} > {args.log_filename.with_suffix("")}.${{SGE_TASK_ID}}-${{LOCAL_PROC_INDEX}}.log 2>&1 &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""
        job_log_filename = args.log_filename.with_suffix(".$TASK_ID.log")
        job_status_filename: Path = args.log_filename.with_suffix(".${SGE_TASK_ID}.out")

        status_filename_list = []
        for array_idx in range(0, args.n_tasks, args.n_tasks_per_proc * args.n_procs):
            status_filename_list.append(args.log_filename.with_suffix(f".{array_idx + 1}.out"))
    else:
        job_log_filename = args.log_filename
        job_status_filename = args.log_filename.with_suffix(".out")

        status_filename_list = [job_status_filename]

    job_script = f"""\
#! /bin/bash

#$-j y
#$-cwd
#$-o {job_log_filename}

set -eE -o pipefail
trap 'echo $? > {job_status_filename}' ERR EXIT  # at error and exit
trap 'echo 143 > {job_status_filename}' TERM  # at termination (by job scheduler)

if [ -n "$PBS_O_WORKDIR" ] && [ "$PBS_ENVIRONMENT" != "PBS_INTERACTIVE" ]; then
    cd $PBS_O_WORKDIR
fi


{config["script_prologue"]}

{job}
"""

    qsub = config["qsub"].format(args=args)
    qsub_args = config[mode]["qsub_args"].format(args=args)

    # Create the job script file, remove old status files, and run the job
    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    job_filename: Path = args.log_filename.with_suffix(".sh")
    with open(job_filename, "w") as f:
        f.write(job_script)

    for status_filename in status_filename_list:
        status_filename.unlink(missing_ok=True)

    subprocess.run(f"{qsub} {qsub_args} {job_filename}", shell=True, check=True)

    for status_filename in status_filename_list:
        while not status_filename.exists():
            time.sleep(1.0)

        status = int(status_filename.read_text())
        if status != 0:
            raise RuntimeError(f"Job failed with {status} exit code.")
        status_filename.unlink()


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
    parent_parser.add_argument("--walltime", type=str, default=config["walltime"])
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
    sub_parser.add_argument("--n_nodes", type=int, default=config["mpi"]["n_nodes"])

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    args = parser.parse_args()
    mode = args.mode + "-array" if getattr(args, "n_tasks", None) is not None else args.mode

    dispatch_job(mode, args, config)


if __name__ == "__main__":
    main()
