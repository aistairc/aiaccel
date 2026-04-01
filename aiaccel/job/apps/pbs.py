#! /usr/bin/env python


# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import shlex

from aiaccel.job.apps import prepare_argument_parser, prepare_job_context, submit_job_and_wait


def main() -> None:
    # Load configuration (from the default YAML string)
    config, parser, sub_parsers = prepare_argument_parser("pbs.yaml")

    args = parser.parse_args()
    mode = args.mode + "-array" if getattr(args, "n_tasks", None) is not None else args.mode

    # Prepare the job script and arguments
    job = config[mode].job.format(command=shlex.join(args.command), args=args)
    job, job_log_filename, job_status_filename, status_filename_list = prepare_job_context(
        args, mode, job, "PBS_ARRAY_INDEX", ".^array_index^.log", ".${PBS_ARRAY_INDEX}.out"
    )

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

    submit_job_and_wait(
        args.log_filename,
        job_script,
        qsub,
        qsub_args,
        status_filename_list,
        bool(config.get("use_scandir", False)),
    )


if __name__ == "__main__":
    main()
