from __future__ import annotations

import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from datetime import datetime
from logging import StreamHandler, getLogger
from pathlib import Path
from typing import Any

import yaml

import aiaccel
from aiaccel.cli import CsvWriter
from aiaccel.common import datetime_format, file_result_csv, resource_type_mpi
from aiaccel.config import load_config
from aiaccel.manager import create_manager
from aiaccel.optimizer import create_optimizer
from aiaccel.storage import Storage
from aiaccel.tensorboard import TensorBoard
from aiaccel.util.buffer import Buffer
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
    manager = create_manager(config.resource.type.value)(config, optimizer)
    tensorboard = TensorBoard(config)
    storage = Storage(workspace.storage_file_path)

    time_s = time.time()
    loop_start_time = datetime.now()
    max_trial_number = config.optimize.trial_number
    end_estimated_time = "Unknown"
    buff = Buffer(["num_finished", "available_pool_size"])
    buff.d["num_finished"].set_max_len(2)
    buff.d["available_pool_size"].set_max_len(2)

    manager.pre_process()

    if config.resource.type.value.lower() == resource_type_mpi and mpi_enable:  # MPI
        Mpi.prepare(workspace.path)

    while True:
        try:
            if not manager.inner_loop_main_process():
                break
            if not manager.is_error_free():
                break
            if int((time.time() - time_s)) % 10 == 0:
                returncodes = storage.returncode.get_all_trial_returncode()
                if any(item != 0 for item in returncodes):
                    logger.error("Some trials are failed.")
                    break
                num_ready, num_running, num_finished = storage.get_num_running_ready_finished()
                available_pool_size = manager.get_available_pool_size(num_ready, num_running, num_finished)
                now = datetime.now()
                looping_time = now - loop_start_time

                if num_finished > 0:
                    one_loop_time = looping_time / num_finished
                    finishing_time = now + (max_trial_number - num_finished) * one_loop_time
                    end_estimated_time = finishing_time.strftime(datetime_format)

                buff.d["num_finished"].Add(num_finished)
                if buff.d["num_finished"].Len == 1 or buff.d["num_finished"].has_difference():
                    manager.logger.info(
                        f"{num_finished}/{max_trial_number} finished, "
                        f"max trial number: {max_trial_number}, "
                        f"ready: {num_ready} ,"
                        f"running: {num_running}, "
                        f"end estimated time: {end_estimated_time}"
                    )

                    # TensorBoard
                    tensorboard.update()

                buff.d["available_pool_size"].Add(available_pool_size)
                if buff.d["available_pool_size"].Len == 1 or buff.d["available_pool_size"].has_difference():
                    manager.logger.info(f"pool_size: {available_pool_size}")

            time.sleep(config.generic.sleep_time)

        except Exception as e:
            logger.exception("Unexpected error occurred.")
            logger.exception(e)
            break

    manager.post_process()
    manager.evaluate()

    csv_writer = CsvWriter(config)
    csv_writer.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()
    if dst is None:
        logger.error("Moving data is failed.")
        return

    config_name = Path(args.config).name
    shutil.copy(Path(args.config), dst / config_name)

    if os.path.exists(workspace.best_result_file):
        with open(workspace.best_result_file, "r") as f:
            final_results: list[dict[str, Any]] = yaml.load(f, Loader=yaml.UnsafeLoader)

        for i, final_result in enumerate(final_results):
            best_id = final_result["trial_id"]
            best_value = final_result["result"][i]
            if best_id is not None and best_value is not None:
                logger.info(f"Best trial [{i}] : {best_id}")
                logger.info(f"\tvalue : {best_value}")
    logger.info(f"result file : {dst}/{file_result_csv}")
    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
