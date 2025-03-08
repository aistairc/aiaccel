from argparse import ArgumentParser
import logging
import os
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.utils import (
    collect_git_status_from_config,
    load_config,
    overwrite_omegaconf_dumper,
    pathlib2str_config,
    print_config,
    print_git_status,
)

logger = logging.getLogger(__name__)


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

    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    parser.add_argument(
        "-s",
        "--submit_job",
        action="store_true",
        help="Submit a training job using the command specified in the config file",
    )
    parser.add_argument(
        "-p",
        "--print_script",
        action="store_true",
        help="Generate the training command from the config file and print it",
    )
    args, unk_args = parser.parse_known_args()

    # load config
    config = oc.merge(
        load_config(
            args.config,
            {
                "config_filename": args.config,
                "working_directory": str(Path(args.config).parent.resolve()),
                "base_config_path": str(Path(__file__).parent / "config"),
            },
        ),
        oc.from_cli(unk_args),
    )

    if args.print_script:
        print(config.train_script)
    elif args.submit_job:
        import subprocess

        status_list = collect_git_status_from_config(config)
        print_git_status(status_list)

        if not all(st.ready() for st in status_list):
            logging.error("There are remaining uncommited file(s).")
        else:
            with open(Path(config.working_directory) / "train.sh", "w") as f:
                f.write(config.train_script)

            subprocess.run(config.job_submission_script.format(job_filename=f.name), shell=True)
    else:
        import lightning as lt

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
