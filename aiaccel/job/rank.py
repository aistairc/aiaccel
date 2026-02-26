# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import os


def get_rank(default: int = 0) -> int:
    for key in [
        "GLOBAL_RANK",  # PyTorch Lightning
        "RANK",  # torchrun / deepspeed / pytorch launcher
        "OMPI_COMM_WORLD_RANK",  # OpenMPI
        "PMI_RANK",  # MPICH / Intel MPI
        "MV2_COMM_WORLD_RANK",  # MVAPICH2
        "SLURM_PROCID",  # Slurm
    ]:
        rank = os.environ.get(key)
        if rank is not None:
            try:
                return int(rank)
            except ValueError:
                pass

    return default


def get_task_list(task_list: list[Any]) -> list[Any]:
    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        return task_list[start:end]
    else:
        return []
