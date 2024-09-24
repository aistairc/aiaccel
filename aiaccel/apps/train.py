from argparse import ArgumentParser
import os
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.utils import print_config


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Config file in YAML format")
    parser.add_argument("--working_directory", type=Path, default=Path.cwd(), help="Working directory")
    args, unk_args = parser.parse_known_args()

    # load config
    config = oc.merge(
        {
            "base_config_path": str(Path(__file__).parent / "config"),
            "base_config": "${base_config_path}/train_base.yaml",
        },
        vars(args),
        oc.load(args.config),
        oc.from_cli(unk_args),
    )

    config = oc.merge(
        oc.load(config.base_config),
        config,
    )

    if "OMPI_COMM_WORLD_RANK" not in os.environ or int(os.environ["OMPI_COMM_WORLD_RANK"]) == 0:
        print_config(config)

        with open(config.working_directory / "config.pkl", "wb") as f:
            pkl.dump(config, f)

    # train
    trainer: lt.Trainer = instantiate(config.trainer)
    trainer.fit(
        model=instantiate(config.task),
        datamodule=instantiate(config.datamodule),
    )


if __name__ == "__main__":
    main()
