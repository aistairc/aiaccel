from __future__ import annotations

import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger

from aiaccel.config import load_config
from pathlib import Path

from aiaccel.cli import CsvWriter
from aiaccel.common import dict_lock, goal_maximize, goal_minimize
from aiaccel.master import create_master
from aiaccel.module import AbstractModule
from aiaccel.optimizer import create_optimizer
from aiaccel.scheduler import create_scheduler
from aiaccel.util import get_file_result_hp, load_yaml
from aiaccel.workspace import Workspace

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())


def get_best_parameter(
    files: list[Path],
    goal: str,
    objective_y_index: int,
    dict_lock: Path
) -> tuple[float | None, Path | None]:
    """Get a best parameter in specified files.

    Args:
        files (list[Path]): A list of files to find a best.
        goal (str): Maximize or Minimize.
        dict_lock (Path): A directory to store lock files.

    Returns:
        tuple[float | None, Path | None]: A best result value and a
        file path. It returns None if a number of files is less than one.

    Raises:
        ValueError: Causes when an invalid goal is set.
    """

    if len(files) < 1:
        return None, None

    yml = load_yaml(files[0], dict_lock)

    try:
        best = float(yml["result"][objective_y_index])
    except TypeError:
        logger = getLogger("root.master.parameter")
        logger.error(f'Invalid result: {yml["result"][objective_y_index]}.')
        return None, None

    best_file = files[0]

    for f in files[1:]:
        yml = load_yaml(f, dict_lock)
        result = float(yml["result"][objective_y_index])

        if goal.lower() == goal_maximize:
            if best < result:
                best, best_file = result, f
        elif goal.lower() == goal_minimize:
            if best > result:
                best, best_file = result, f
        else:
            logger = getLogger("root.master.parameter")
            logger.error(f"Invalid goal: {goal}.")
            raise ValueError(f"Invalid goal: {goal}.")

    return best, best_file


def main() -> None:  # pragma: no cover
    """Parses command line options and executes optimization."""
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--clean', nargs='?', const=True, default=False)
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        logger.error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return

    config.resume = args.resume
    config.clean = args.clean

    workspace = Workspace(config.generic.workspace)
    goals = [item.value for item in config.optimize.goal]
    path_to_lock_file = workspace.path / dict_lock

    if config.resume is None:
        if config.clean is True:
            logger.info("Cleaning workspace")
            workspace.clean()
            logger.info(f"Workspace directory {str(workspace.path)} is cleaned.")
        else:
            if workspace.exists():
                logger.info("workspace exists.")
                return

    workspace.create()
    if workspace.check_consists() is False:
        logger.error("Creating workspace is Failed.")
        return

    logger.info(f"config: {str(pathlib.Path(config.config_path).resolve())}")

    Master = create_master(config.resource.type.value)
    Optimizer = create_optimizer(config.optimize.search_algorithm)
    Scheduler = create_scheduler(config.resource.type.value)
    modules: list[AbstractModule] = [Master(config), Optimizer(config), Scheduler(config)]

    time_s = time.time()

    for module in modules:
        module.pre_process()

    while True:
        for module in modules:
            if not module.inner_loop_main_process():
                break
            if not module.check_error():
                break
            module.loop_count += 1
        else:
            time.sleep(config.generic.sleep_time)
            continue
        break

    for module in modules:
        module.post_process()

    csv_writer = CsvWriter(config)
    csv_writer.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()
    if dst is None:
        logger.error("Moving data is failed.")
        return

    config_name = Path(args.config).name
    shutil.copy(Path(args.config), dst / config_name)

    files = get_file_result_hp(dst)

    for i in range(len(goals)):
        best, best_file = get_best_parameter(files, goals[i], i, path_to_lock_file)
        logger.info(f"Best result [{i}] : {best_file}")
        logger.info(f"\tvalue : {best}")

    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
