from __future__ import annotations

import os
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger
from pathlib import Path

from aiaccel.cli import CsvWriter
from aiaccel.config import Config
from aiaccel.master import create_master
from aiaccel.module import AbstractModule
from aiaccel.optimizer import create_optimizer
from aiaccel.scheduler import create_scheduler
from aiaccel.storage.storage import Storage
from aiaccel.workspace import Workspace

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())


def main() -> None:  # pragma: no cover
    """Parses command line options and executes optimization."""
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--clean', nargs='?', const=True, default=False)
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
    modules: list[AbstractModule] = [Master(vars(args)), Optimizer(vars(args)), Scheduler(vars(args))]

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

    if best_results is None:
        logger.info("No result.")
        return

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
