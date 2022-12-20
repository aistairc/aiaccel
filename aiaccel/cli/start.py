import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger

import aiaccel
from aiaccel import parameter as pt
from aiaccel.config import load_config
from aiaccel.master.create import create_master
from aiaccel.optimizer.create import create_optimizer
from aiaccel.scheduler.create import create_scheduler
from aiaccel.util import filesystem as fs
from aiaccel.util.report import CreationReport
from aiaccel.workspace import Workspace

logger = getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.addHandler(StreamHandler())


def main() -> None:  # pragma: no cover
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--clean', nargs='?', const=True, default=False)
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        logger.error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return

    config.config_path = args.config
    config.resume = args.resume
    config.clean = args.clean

    workspace = Workspace(config.generic.workspace)
    dict_lock = workspace.path / aiaccel.dict_lock

    if config.resume is None:
        if config.clean is True:
            logger.info("Cleaning workspace")
            workspace.clean()
            logger.info(f'Workspace directory {str(workspace.path)} is cleaned.')
        else:
            if workspace.exists():
                logger.info("workspace exists.")
                return

    workspace.create()
    if workspace.check_consists() is False:
        logger.error("Creating workspace is Failed.")
        return

    logger.info(f"config: {str(pathlib.Path(config.config_path).resolve())}")

    Master = create_master(config.resource.type)
    Optimizer = create_optimizer(config.optimize.search_algorithm)
    Scheduler = create_scheduler(config.resource.type)
    modules = [Master(config), Optimizer(config), Scheduler(config)]

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

    report = CreationReport(config)
    report.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()

    config_name = pathlib.Path(config.config_path).name
    shutil.copy(pathlib.Path(config.config_path), dst / config_name)

    files = fs.get_file_result_hp(dst)
    best, best_file = pt.get_best_parameter(files, config.optimize.goal, dict_lock)

    logger.info(f"Best result    : {best_file}")
    logger.info(f"               : {best}")
    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
