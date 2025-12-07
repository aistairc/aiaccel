# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import logging
import os

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.config import (
    prepare_config,
)
from aiaccel.config.git import collect_git_status_from_config, print_git_status

logger = logging.getLogger(__name__)


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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    is_rank_zero = get_rank() == 0
    config = prepare_config(
        config_filename=args.config,
        overwrite_config=oc.from_cli(unk_args),
        print_config=is_rank_zero,
        save_config=is_rank_zero,
        save_filename="merged_config.yaml",
    )

    if is_rank_zero:
        status_list = collect_git_status_from_config(config)
        print_git_status(status_list)

    if "seed" in config:
        lt.seed_everything(config.seed, workers=True)

    # build trainer
    trainer: lt.Trainer = instantiate(config.trainer)

    # start training
    trainer.fit(
        model=instantiate(config.task),
        datamodule=instantiate(config.datamodule),
    )


if __name__ == "__main__":
    main()
