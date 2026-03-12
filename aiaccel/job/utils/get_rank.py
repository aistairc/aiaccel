# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import os


def get_rank(default: int = 0) -> int:
    for key in [
        "LOCAL_RANK",  # PyTorch Lightning
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
