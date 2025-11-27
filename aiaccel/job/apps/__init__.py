from argparse import ArgumentParser, _SubParsersAction
from importlib import resources
import os
from pathlib import Path

from omegaconf import DictConfig

from aiaccel.config import load_config, resolve_inherit, setup_omegaconf

setup_omegaconf()


def prepare_argument_parser(
    default_config_name: str,
) -> tuple[DictConfig, ArgumentParser, _SubParsersAction]:  # type: ignore
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args()

    args.config = Path(
        args.config
        or os.environ.get("AIACCEL_JOB_CONFIG")
        or (Path(str(resources.files(__package__) / "config")) / default_config_name)
    )  # type: ignore

    config: DictConfig = load_config(args.config, is_print_config=args.print_config)  # type: ignore

    parser = ArgumentParser()
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--config", type=Path)
    sub_parsers = parser.add_subparsers(dest="mode", required=True)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--walltime", type=str, default=config.walltime)
    parent_parser.add_argument("log_filename", type=Path)
    parent_parser.add_argument("command", nargs="+")

    sub_parser = sub_parsers.add_parser("cpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["cpu-array"].n_tasks_per_proc)
    sub_parser.add_argument("--n_procs", type=int, default=config["cpu-array"].n_procs)

    sub_parser = sub_parsers.add_parser("gpu", parents=[parent_parser])
    sub_parser.add_argument("--n_tasks", type=int)
    sub_parser.add_argument("--n_tasks_per_proc", type=int, default=config["gpu-array"].n_tasks_per_proc)
    sub_parser.add_argument("--n_procs", type=int, default=config["gpu-array"].n_procs)

    sub_parser = sub_parsers.add_parser("mpi", parents=[parent_parser])
    sub_parser.add_argument("--n_procs", type=int, required=True)
    sub_parser.add_argument("--n_nodes", type=int, default=config["mpi"].n_nodes)

    sub_parser = sub_parsers.add_parser("train", parents=[parent_parser])
    sub_parser.add_argument("--n_gpus", type=int)

    return config, parser, sub_parsers
