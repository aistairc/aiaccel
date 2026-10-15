# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import os


def get_rank(default: int = 0) -> int:
    """Return the process rank obtained from a supported environment variable.

    The environment variables are checked in the following order:
    ``GLOBAL_RANK``, ``RANK``, ``OMPI_COMM_WORLD_RANK``, ``PMI_RANK``,
    ``MV2_COMM_WORLD_RANK``, and ``SLURM_PROCID``. Variables whose values
    cannot be converted to integers are ignored.

    Args:
        default (int, optional): Value to return when none of the supported
            environment variables contains a valid integer. Defaults to 0.

    Returns:
        int: The first valid process rank found, or ``default`` if none is
        available.
    """

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
