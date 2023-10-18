from __future__ import annotations

import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger
from pathlib import Path
from typing import Any

import yaml

import aiaccel
from aiaccel.cli import CsvWriter
from aiaccel.common import dict_result, extension_hp, resource_type_mpi
from aiaccel.config import load_config
from aiaccel.module import AbstractModule
from aiaccel.optimizer import create_optimizer
from aiaccel.scheduler import create_scheduler
from aiaccel.tensorboard import TensorBoard
from aiaccel.workspace import Workspace

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())

mpi_enable = aiaccel.util.mpi.mpi_enable
if mpi_enable:
    from aiaccel.util.mpi import Mpi


def main() -> None:  # pragma: no cover
    """Parses command line options and executes optimization."""
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yml")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--clean", nargs="?", const=True, default=False)

    parser.add_argument("--from_mpi_bat", action="store_true", help="Only aiaccel is used when mpi bat.")
    parser.add_argument("--make_hostfile", action="store_true", help="Only aiaccel is used when mpi bat.")
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        logger.error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return

    if config.resource.type.value.lower() == resource_type_mpi:  # MPI
        if not mpi_enable:
            raise Exception("MPI is not enabled.")
        if args.make_hostfile:
            Mpi.make_hostfile(config, logger)
            return
        if not args.from_mpi_bat:
            Mpi.run_bat(config, logger)
            return
        logger.info("MPI is enabled.")
        if Mpi.gpu_max == 0:
            Mpi.gpu_max = config.resource.mpi_npernode
        Mpi.run_main()

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

    optimizer = create_optimizer(config.optimize.search_algorithm)(config)
    scheduler = create_scheduler(config.resource.type.value)(config)
    tensorboard = TensorBoard(config)
    modules: list[AbstractModule] = [optimizer, scheduler, tensorboard]

    time_s = time.time()

    for module in modules:
        module.pre_process()

    if config.resource.type.value.lower() == resource_type_mpi and mpi_enable:  # MPI
        Mpi.prepare(workspace.path)

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

    scheduler.evaluate()

    csv_writer = CsvWriter(config)
    csv_writer.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()
    if dst is None:
        logger.error("Moving data is failed.")
        return

    config_name = Path(args.config).name
    shutil.copy(Path(args.config), dst / config_name)

    if os.path.exists(workspace.final_result_file):
        with open(workspace.final_result_file, "r") as f:
            final_results: list[dict[str, Any]] = yaml.load(f, Loader=yaml.UnsafeLoader)

        for i, final_result in enumerate(final_results):
            best_id = final_result["trial_id"]
            best_value = final_result["result"][i]
            if best_id is not None and best_value is not None:
                logger.info(f"Best result [{i}] : {dst}/{dict_result}/{best_id}.{extension_hp}")
                logger.info(f"\tvalue : {best_value}")

    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
