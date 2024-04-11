from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc

import lightning as lt
from lightning.fabric.utilities.rank_zero import rank_zero_only

from aiaccel.utils import print_config


@rank_zero_only
def print_and_save_config(config: DictConfig | ListConfig) -> None:
    print_config(config)

    with open(config.working_directory / "config.pkl", "wb") as f:
        pkl.dump(config, f)


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

    print_and_save_config(config)

    # train
    trainer: lt.Trainer = instantiate(config.trainer)
    trainer.fit(
        model=instantiate(config.task),
        datamodule=instantiate(config.datamodule),
    )


if __name__ == "__main__":
    main()
