# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import logging
import warnings

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.config import (
    prepare_config,
)
from aiaccel.config.git import collect_git_status_from_config, print_git_status
from aiaccel.job.utils import get_rank

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    parser.add_argument("--seed_distributed", action="store_true")
    args, unk_args = parser.parse_known_args()

    is_rank_zero = (rank := get_rank()) == 0
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
        if args.seed_distributed:
            lt.seed_everything(config.seed + rank, workers=True)
        else:
            if rank != 0:
                warnings.warn(
                    "DDP may be running without '--seed_distributed' option being specified. "
                    "This feature is planned to be integrated into seed in the future.",
                    stacklevel=2,
                )
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
