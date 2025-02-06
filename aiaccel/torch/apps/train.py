from argparse import ArgumentParser
import os
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

import lightning as lt

from aiaccel.utils import load_config, pathlib2str_config, print_config


def main() -> None:
    """
    Execute the training process using a configuration file.

    This function:
    1. Parses command-line arguments to get the configuration file and working directory.
    2. Loads and merges configurations from the YAML file, command-line arguments, and default settings.
    3. Prints the final configuration.
    4. Saves the merged configuration to a file (`${working_directory}/config_merged.yaml`).
    5. Instantiates and runs the training process using the specified trainer, model, and datamodule.

    Command-line Arguments:
        config (Path): Path to the YAML configuration file.
        --working_directory (Path, optional): Path to the working directory (default: current working directory).

    Usage Example:
        ```bash
        python -m aiaccel.torch.apps.train config.yaml --working_directory /path/to/working/directory
        ```
        You can also update some configurations from CLI as follows:
        ```bash
        python -m aiaccel.torch.apps.train config.yaml task.hparam1=1.0 task.hparam2=2.0 ...
        ```
    """

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    parser.add_argument("--working_directory", type=str, default=str(Path.cwd()), help="Working directory")
    args, unk_args = parser.parse_known_args()

    # load config
    config = load_config(
        args.config,
        {
            "base_config_path": str(Path(__file__).parent / "config"),
            "base_config": "${base_config_path}/train_base.yaml",
        }
        | vars(args),
    )
    config = oc.merge(config, oc.from_cli(unk_args))

    config = oc.merge(
        oc.load(config.base_config),
        config,
    )

    if int(os.environ.get("OMPI_COMM_WORLD_RANK", 0)) == 0 and int(os.environ.get("RANK", 0)) == 0:
        print_config(config)

    # build trainer
    trainer: lt.Trainer = instantiate(config.trainer)

    # save config
    if trainer.is_global_zero:
        if "merged_config_path" in config:
            merged_config_path = config.merged_config_path
        else:
            merged_config_path = Path(config.working_directory) / "config_merged.yaml"

        with open(merged_config_path, "w") as f:
            oc.save(pathlib2str_config(config), f)

    # start training
    trainer.fit(
        model=instantiate(config.task),
        datamodule=instantiate(config.datamodule),
    )


if __name__ == "__main__":
    main()
