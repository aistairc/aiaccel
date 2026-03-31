#! /usr/bin/env python


# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import shlex

from aiaccel.job.apps import _submit_job_and_wait, prepare_argument_parser


def main() -> None:
    # Load configuration (from the default YAML string)
    config, parser, sub_parsers = prepare_argument_parser("pbs.yaml")

    args = parser.parse_args()
    mode = args.mode + "-array" if getattr(args, "n_tasks", None) is not None else args.mode

    # Prepare the job script and arguments
    job = config[mode].job.format(command=shlex.join(args.command), args=args)

    if mode in ["cpu-array", "gpu-array"]:
        job = f"""\
for LOCAL_PROC_INDEX in {{1..{args.n_procs}}}; do
    TASK_INDEX=$(( PBS_ARRAY_INDEX + {args.n_tasks_per_proc} * (LOCAL_PROC_INDEX - 1) ))

    if [[ $TASK_INDEX -gt {args.n_tasks} ]]; then
        break
    fi

    TASK_INDEX=$TASK_INDEX \\
    TASK_STEPSIZE={args.n_tasks_per_proc} \\
        {job} > {args.log_filename.with_suffix("")}.${{PBS_ARRAY_INDEX}}-${{LOCAL_PROC_INDEX}}.log 2>&1 &

    pids[$LOCAL_PROC_INDEX]=$!
done

for i in "${{!pids[@]}}"; do
    wait ${{pids[$i]}}
done
"""
        job_log_filename = args.log_filename.with_suffix(".^array_index^.log")
        job_status_filename: Path = args.log_filename.with_suffix(".${PBS_ARRAY_INDEX}.out")

        status_filename_list = []
        for array_idx in range(0, args.n_tasks, args.n_tasks_per_proc * args.n_procs):
            status_filename_list.append(args.log_filename.with_suffix(f".{array_idx + 1}.out"))
    else:
        job_log_filename = args.log_filename
        job_status_filename = args.log_filename.with_suffix(".out")

        status_filename_list = [job_status_filename]

    job_script = f"""\
#! /bin/bash

#PBS -j oe
#PBS -k oed
#PBS -o {job_log_filename}

set -eE -o pipefail
trap 'echo $? > {job_status_filename}' ERR EXIT  # at error and exit
trap 'echo 143 > {job_status_filename}' TERM  # at termination (by job scheduler)

if [ -n "$PBS_O_WORKDIR" ] && [ "$PBS_ENVIRONMENT" != "PBS_INTERACTIVE" ]; then
    cd $PBS_O_WORKDIR
fi


{config.script_prologue}

{job}
"""

    qsub = config.qsub.format(args=args)
    qsub_args = config[mode].qsub_args.format(args=args)

    _submit_job_and_wait(
        args.log_filename,
        job_script,
        qsub,
        qsub_args,
        status_filename_list,
        bool(config.get("use_scandir", False)),
    )


if __name__ == "__main__":
    main()
