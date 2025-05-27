#! /usr/bin/env python

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
import shlex
import subprocess
import time


def prepare_mpi_job(args: Namespace, command: str) -> tuple[str, list[str], list[Path]]:
    lock_filename = args.log_filename.with_suffix(".lock")

    job = f"""\
source /etc/profile.d/modules.sh
module load hpcx

mpirun -bind-to none -map-by slot \\
    -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
    -x SINGULARITYENV_OPAL_PREFIX=/usr/local/ \\
    -x SINGULARITYENV_PMIX_INSTALL_PREFIX=/usr/local/ \\
    {command}

rm {lock_filename}\
"""

    n_mpiprocs = args.n_procs // args.n_nodes
    n_ompthreads = (args.n_nodes * 96) // args.n_procs
    qsub_args = ["-l", f"select={args.n_nodes}:mpiprocs={n_mpiprocs}:ompthreads={n_ompthreads}"]
    qsub_args += ["-o", str(args.log_filename)]

    return job, qsub_args, [lock_filename]


def prepare_single_job(args: Namespace, command: str) -> tuple[str, list[str], list[Path]]:
    lock_filename = args.log_filename.with_suffix(".lock")

    job = f"{command} && rm {lock_filename}"

    qsub_args = ["-l", "select=1"]
    qsub_args += ["-o", str(args.log_filename)]

    return job, qsub_args, [lock_filename]


def prepare_array_jobs(args: Namespace, command: str) -> tuple[str, list[str], list[Path]]:
    assert args.n_tasks_per_proc % args.n_procs_per_job == 0

    n_tasks_per_proc = args.n_tasks_per_proc // args.n_procs_per_job

    job = f"""\
for LOCAL_PROC_INDEX in {{1..{args.n_procs_per_job}}}; do
    TASK_INDEX=$(( PBS_ARRAY_INDEX + {n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {args.n_tasks} ]]; then
        break
    fi

    CUDA_VISIBLE_DEVICES=$(( LOCAL_PROC_INDEX % 8 )) \\
    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={n_tasks_per_proc} \\
        {command} > {args.log_filename.with_suffix(".${PBS_ARRAY_INDEX}.proc-${LOCAL_PROC_INDEX}.log")} 2>&1 \\
        && rm {args.log_filename.with_suffix(".${PBS_ARRAY_INDEX}.proc-${LOCAL_PROC_INDEX}.lock")} &
done
wait\
"""

    qsub_args = ["-l", "select=1"]
    qsub_args += ["-J", f"1-{args.n_tasks}:{args.n_tasks_per_proc}"]
    qsub_args += ["-o", str(args.log_filename.with_suffix(".^array_index^.log"))]

    lock_filename_list = []
    for array_idx in range(0, args.n_tasks, args.n_tasks_per_proc):
        for local_proc_idx in range(0, args.n_procs_per_job):
            task_index = array_idx + n_tasks_per_proc * local_proc_idx + 1
            if task_index >= args.n_tasks:
                break

            lock_filename_list.append(args.log_filename.with_suffix(f".{array_idx + 1}.proc-{local_proc_idx + 1}.lock"))

    return job, qsub_args, lock_filename_list


def prepare_train_job(args: Namespace, command: str) -> tuple[str, list[str], list[Path]]:
    lock_filename = args.log_filename.with_suffix(".lock")

    job = f"""\
source /etc/profile.d/modules.sh
module load hpcx

mpirun -bind-to none -map-by slot \\
    -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
    -x MAIN_ADDR=$(hostname -i) \\
    -x MAIN_PORT=3000 \\
    -x COLUMNS=120 \\
    -x PYTHONUNBUFFERED=true \\
    {command} && \\
rm {lock_filename}\
"""

    n_nodes = args.n_gpus // 8
    qsub_args = ["-l", f"select={n_nodes}:mpiprocs=8:ompthreads=12"]
    qsub_args += ["-o", str(args.log_filename)]

    return job, qsub_args, [lock_filename]


def main() -> None:  # noqa: C901
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--local", action="store_true")
    parent_parser.add_argument("--use_singularity", action="store_true")
    parent_parser.add_argument("--walltime", type=str, default="0:40:0")
    parent_parser.add_argument("log_filename", type=Path)
    parent_parser.add_argument("command", nargs="+")

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

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    args = parser.parse_args()

    args.log_filename.parent.mkdir(exist_ok=True, parents=True)

    if args.use_singularity:
        # singularity
        singularity_path = Path(__file__).resolve().parent.parent.parent / "singularity" / "singularity.sif"
        command = f"singularity exec --nv {singularity_path} direnv exec . {shlex.join(args.command)}"
    else:
        # venv
        venv_path = "./env"
        command = f"source {venv_path}/bin/activate && {shlex.join(args.command)}"

    if args.mode in ["cpu", "gpu"]:
        if args.n_tasks is None:
            job, qsub_args, lock_filename_list = prepare_single_job(args, command)
        else:
            job, qsub_args, lock_filename_list = prepare_array_jobs(args, command)
    elif args.mode == "mpi":
        job, qsub_args, lock_filename_list = prepare_mpi_job(args, command)
    elif args.mode == "train":
        job, qsub_args, lock_filename_list = prepare_train_job(args, command)
    else:
        raise ValueError()

    for lock_filename in lock_filename_list:
        lock_filename.touch()

    job_filename: Path = args.log_filename.with_suffix(".sh")
    status_filename: Path = args.log_filename.with_suffix(".out")
    with open(job_filename, "w") as f:
        f.write(
            f"""\
#! /bin/bash

#PBS -j oe
#PBS -k oed

set -e

trap 'echo $? > {status_filename}' ERR

if [ -n "$PBS_O_WORKDIR" ] && [ "$PBS_ENVIRONMENT" != "PBS_INTERACTIVE" ]; then
    cd $PBS_O_WORKDIR
fi

echo Job ID: $PBS_JOBID
echo Hostname: $(hostname)

export NVIDIA_VISIBLE_DEVICES=all
export SINGULARITY_BINDPATH="$SINGULARITY_BINDPATH,/groups,/groups-2.0"

export SINGULARITYENV_COLUMNS=120
export SINGULARITYENV_PYTHONUNBUFFERED=true

{job}
"""
        )

    if not args.local:
        qsub_command = ["qsub", "-P", os.environ["JOB_GROUP"], "-q", "rt_HF"]
        qsub_command += ["-l", f"walltime={args.walltime}", "-v", "USE_SSH=1"]
        qsub_command += qsub_args + [str(job_filename)]

        qsub_command_str = shlex.join(qsub_command)

        subprocess.run(qsub_command_str, shell=True, check=True)

        for lock_filename in lock_filename_list:
            while lock_filename.exists():
                time.sleep(1.0)
            if status_filename.exists():
                raise RuntimeError("Job failed")
    else:
        subprocess.run(["bash", str(job_filename)], check=True)


if __name__ == "__main__":
    main()
