import os
from pathlib import Path
from argparse import ArgumentParser

from omegaconf import OmegaConf as oc
from hydra.utils import instantiate

import lightning as pl

import aiaccel
from aiaccel.utils import print_config


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        description="Config file in YAML format",
    )
    parser.add_argument(
        "--working_directory",
        type=Path,
        default=Path.cwd(),
        description="Working directory",
    )
    parser.add_argument(
        "--base_config",
        type=Path,
        default=Path(aiaccel.__path__) / "apps" / "cfg" / "train.yaml",
        description="Base config file",
    )
    args, unk_args = parser.parse_known_args()

    # load config
    config = oc.merge(
        oc.load(args.base_config),
        oc.load(args.config),
        vars(args),
        oc.from_cli(unk_args),
    )

    if "OMPI_COMM_WORLD_RANK" not in os.environ or int(os.environ["OMPI_COMM_WORLD_RANK"]) == 0:
        print_config(config)

    # train
    trainer: pl.Trainer = instantiate(config.trainer)
    trainer.fit(
        model=instantiate(config.model),
        datamodule=instantiate(config.datamodule),
    )


if __name__ == "__main__":
    main()
