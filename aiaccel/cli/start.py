from __future__ import annotations

import os
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger
from pathlib import Path

from aiaccel.cli import CsvWriter
from aiaccel.common import goal_maximize, goal_minimize
from aiaccel.config import Config
from aiaccel.master import create_master
from aiaccel.optimizer import create_optimizer
from aiaccel.scheduler import create_scheduler
from aiaccel.storage.storage import Storage
from aiaccel.util import load_yaml
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
    parser.add_argument("--config", "-c", type=str, default="config.yml")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--clean", nargs="?", const=True, default=False)
    args = parser.parse_args()

    config = Config(args.config, warn=True, format_check=True)
    if config is None:
        logger.error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return

    workspace = Workspace(config.workspace.get())

    if args.resume is None:
        if args.clean is True:
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

    storage = Storage(workspace.path)

    logger.info(f"config: {str(Path(args.config).resolve())}")

    Master = create_master(args.config)
    Optimizer = create_optimizer(args.config)
    Scheduler = create_scheduler(args.config)
    modules = [Master(vars(args)), Optimizer(vars(args)), Scheduler(vars(args))]

    sleep_time = config.sleep_time.get()
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
            time.sleep(sleep_time)
            continue
        break

    for module in modules:
        module.post_process()

    csv_writer = CsvWriter(args.config)
    csv_writer.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()
    if dst is None:
        logger.error("Moving data is failed.")
        return

    if isinstance(config.goal.get(), str):
        best_results = storage.get_best_trial_dict([config.goal.get()])
    else:
        best_results = storage.get_best_trial_dict(config.goal.get())

    config_name = Path(args.config).name
    shutil.copy(Path(args.config), dst / config_name)

    for i in range(len(best_results)):
        parametets = str([f'{param["parameter_name"]}: {param["value"]}' for param in best_results[i]["parameters"]])
        parametets = parametets.replace("[", "")
        parametets = parametets.replace("]", "")
        parametets = parametets.replace("'", "")
        logger.info(f"Best result[{i}]")
        logger.info(f"- trial id  : {best_results[i]['trial_id']}")
        logger.info(f"- parameter : {parametets}")
        logger.info(f"- result    : {best_results[i]['result'][i]}")
    logger.info(f"File: {dst.resolve()}")
    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")

    return


if __name__ == "__main__":  # pragma: no cover
    main()
