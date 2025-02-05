from argparse import ArgumentParser
import os
from pathlib import Path
import pickle as pkl

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.utils import print_config


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

    with initialize_config_dir(version_base=None, config_dir=str(Path(args.config).parent.resolve())):
        user_config = compose(config_name=Path(args.config).stem)

    # load config
    config = oc.merge(
        {
            "base_config_path": str(Path(__file__).parent / "config"),
            "base_config": "${base_config_path}/train_base.yaml",
        },
        vars(args),
        user_config,
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
