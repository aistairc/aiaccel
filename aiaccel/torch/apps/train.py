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
    print_config,
)
from aiaccel.config.git import collect_git_status_from_config, print_git_status

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    config, raw_config = load_config(
        config_filename=args.config,
        overwrite_config=oc.from_cli(unk_args),
    )

    if int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)) == 0 and int(os.environ.get("RANK", 0)) == 0:
        print_config(config)
        status_list = collect_git_status_from_config(raw_config)
        print_git_status(status_list)

    # build trainer
    trainer: lt.Trainer = instantiate(raw_config.trainer)

    # save config
    if trainer.is_global_zero:
        Path(raw_config.working_directory).mkdir(parents=True, exist_ok=True)
        merged_config_path = Path(raw_config.working_directory) / "merged_config.yaml"

        with open(merged_config_path, "w") as f:
            oc.save(pathlib2str_config(raw_config), f)

    # start training
    trainer.fit(
        model=instantiate(raw_config.task),
        datamodule=instantiate(raw_config.datamodule),
    )


if __name__ == "__main__":
    main()
