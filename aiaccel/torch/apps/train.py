from argparse import ArgumentParser
import os
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.utils import print_config


def load_user_config(config: Path) -> DictConfig | ListConfig:
    user_config = oc.load(config)
    if isinstance(user_config, DictConfig) and "_base_" in user_config:
        base_config = load_user_config(Path(user_config["_base_"]))
        merge_user_config = oc.merge(base_config, user_config)
        return merge_user_config
    else:
        return user_config


def main() -> None:
    """
    Main function to execute the training process.
    This function parses command-line arguments to get the configuration file
    and working directory, loads and merges configuration settings from various
    sources, prints the configuration if the process is the main one, saves the
    configuration to a file, and then initiates the training process using the
    specified trainer, model, and datamodule.
    Command-line Arguments:
        config (Path): Path to the configuration file in YAML format.
        --working_directory (Path): Path to the working directory (default is the current working directory).
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        Exception: If there is an error during the training process.

    Usage:
        python -m aiaccel.torch.apps.train train.yaml --working_directory /path/to/working/directory
    """

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
        load_user_config(args.config),
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
