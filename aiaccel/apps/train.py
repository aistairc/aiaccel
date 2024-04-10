import os
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf as oc

from aiaccel.utils import print_config


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Config file in YAML format",
    )
    parser.add_argument(
        "--working_directory",
        type=str,
        default=str(Path.cwd()),  # todo: treat path as Path
        help="Working directory",
    )
    args, unk_args = parser.parse_known_args()

    # load config
    config = oc.merge(
        oc.load(args.config),
        vars(args),
        oc.from_cli(unk_args),
    )

    if "OMPI_COMM_WORLD_RANK" not in os.environ or int(os.environ["OMPI_COMM_WORLD_RANK"]) == 0:
        print_config(config)

        with open(Path(config.working_directory) / "config.pkl", "wb") as f:
            pkl.dump(config, f)

    # train
    trainer: pl.Trainer = instantiate(config.trainer)
    trainer.fit(
        model=instantiate(config.task),
        datamodule=instantiate(config.datamodule),
    )


if __name__ == "__main__":
    main()
