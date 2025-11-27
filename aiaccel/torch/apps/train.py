from argparse import ArgumentParser
import logging
import os
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.config import (
    load_config,
    pathlib2str_config,
)
from aiaccel.config.git import collect_git_status_from_config, print_git_status

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    is_rank_zero = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)) == 0 and int(os.environ.get("RANK", 0)) == 0

    config = load_config(
        config_filename=args.config,
        overwrite_config=oc.from_cli(unk_args),
        is_print_config=is_rank_zero,
    )

    if is_rank_zero:
        status_list = collect_git_status_from_config(config)
        print_git_status(status_list)

    # build trainer
    trainer: lt.Trainer = instantiate(config.trainer)

    # save config
    if trainer.is_global_zero:
        Path(config.working_directory).mkdir(parents=True, exist_ok=True)
        merged_config_path = Path(config.working_directory) / "merged_config.yaml"

        with open(merged_config_path, "w") as f:
            oc.save(pathlib2str_config(config), f)

    # start training
    trainer.fit(
        model=instantiate(config.task),
        datamodule=instantiate(config.datamodule),
    )


if __name__ == "__main__":
    main()
