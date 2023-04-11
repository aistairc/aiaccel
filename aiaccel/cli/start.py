from __future__ import annotations
import yaml
from typing import Any

import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger

from aiaccel.config import load_config
from pathlib import Path

from aiaccel.cli import CsvWriter
from aiaccel.master import create_master
from aiaccel.module import AbstractModule
from aiaccel.optimizer import create_optimizer
from aiaccel.scheduler import create_scheduler
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

    config = load_config(args.config)
    if config is None:
        logger.error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return

    config.resume = args.resume
    config.clean = args.clean

    workspace = Workspace(config.generic.workspace)

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

    with open(workspace.final_result_file, 'r') as f:
        final_results: list[dict[str, Any]] = yaml.load(f, Loader=yaml.UnsafeLoader)

    for i, final_result in enumerate(final_results):
        best_id = final_result["trial_id"]
        best_value = final_result["result"][i]
        if best_id is not None and best_value is not None:
            logger.info(f"Best result ID [{i}] : {best_id}")
            logger.info(f"\tvalue : {best_value}")

    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
