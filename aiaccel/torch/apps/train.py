from argparse import ArgumentParser
import logging
import os
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.config import (
    load_config,
    overwrite_omegaconf_dumper,
    pathlib2str_config,
    print_config,
    resolve_inherit,
)

logger = logging.getLogger(__name__)


def main() -> None:
    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    # load config
    config = oc.merge(
        load_config(
            args.config,
            {
                "config_path": args.config,
                "working_directory": str(Path(args.config).parent.resolve()),
                "base_config_path": str(Path(__file__).parent / "config"),
            },
        ),
        oc.from_cli(unk_args),
    )

    if int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)) == 0 and int(os.environ.get("RANK", 0)) == 0:
        print_config(config)

    config = resolve_inherit(config)

    # build trainer
    trainer: lt.Trainer = instantiate(config.trainer)

    # save config
    if trainer.is_global_zero:
        if "merged_config_path" in config:
            merged_config_path = config.merged_config_path
        else:
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
