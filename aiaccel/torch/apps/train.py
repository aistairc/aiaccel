# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT


from argparse import ArgumentParser
import logging
import warnings

from hydra.utils import instantiate
from omegaconf import DictConfig
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

    rank = get_rank()
    is_rank_zero = rank == 0
    config = prepare_config(
        config_filename=args.config,
        overwrite_config=oc.from_cli(unk_args),
        print_config=is_rank_zero,
        save_config=is_rank_zero,
        save_filename="merged_config.yaml",
    )
    assert isinstance(config, DictConfig)

    if is_rank_zero:
        status_list = collect_git_status_from_config(config)
        print_git_status(status_list)

    if "seed" in config:
        if config.get("seed_ddp_mode", False):
            lt.seed_everything(config.seed + rank, workers=True)
        else:
            lt.seed_everything(config.seed, workers=True)

            if rank != 0:
                warnings.warn(
                    "'seed' currently uses the same random seed on all DDP ranks. "
                    "This behavior is discouraged because it can lead to identical RNG streams across processes. "
                    "For distributed runs, use 'seed_distributed' for now. "
                    "In a future release, the behavior of 'seed' will change to match the current "
                    "'seed_distributed' behavior.",
                    FutureWarning,
                    stacklevel=2,
                )

    # build trainer
    trainer: lt.Trainer = instantiate(config.trainer)

    # start training
    trainer.fit(
        model=instantiate(config.task),
        datamodule=instantiate(config.datamodule),
        **config.get("fit_args", {}),
    )


if __name__ == "__main__":
    main()
