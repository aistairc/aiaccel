from logging import getLogger, StreamHandler, DEBUG

import pathlib
import shutil
import time

import aiaccel
from aiaccel import parameter as pt
from aiaccel.argument import Arguments
from aiaccel.config import Config
from aiaccel.master.create import create_master
from aiaccel.optimizer.create import create_optimizer
from aiaccel.scheduler.create import create_scheduler
from aiaccel.util import filesystem as fs
from aiaccel.util.report import CreationReaport
from aiaccel.workspace import Workspace


def main(options: dict = Arguments()) -> None:

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.addHandler(StreamHandler())

    if options['config'] == "" or options['config'] is None:
        logger.info(
            "The configuration file is not specified. "
            "Please specify it with the command line argument "
            "'--config' or '-c'."
        )
        return

    if not pathlib.Path(options['config']).exists():
        logger.info(f"config file: {options['config']} doesn't exist.")
        return

    config = Config(options['config'], warn=True, format_check=True)
    if config is None:
        logger.info(f"Invalid workspace: {options['workspace']} or config: {options['config']}")
        return

    workspace = Workspace(config.workspace.get())
    goal = config.goal.get()
    dict_lock = workspace.path / aiaccel.dict_lock

    if options['resume'] is None:
        if options['clean'] is True:
            logger.info("Cleaning workspace")
            workspace.clean()
            logger.info(f'Workspace directory {str(workspace.path)} is cleaned.')
        else:
            if workspace.exists():
                logger.info("workspace exists.")
                return

    workspace.create()
    if workspace.check_consists() is False:
        logger.info("Creating workspace is Failed.")
        return

    logger.info(f"Start {config.search_algorithm.get()} Optimization")
    logger.info(f"config: {str(pathlib.Path(options['config']).resolve())}")

    time_s = time.time()

    Master = create_master(options['config'])
    Optimizer = create_optimizer(options['config'])
    Scheduler = create_scheduler(options['config'])
    modules = [Master(options), Scheduler(options), Optimizer(options)]

    sleep_time = config.sleep_time.get()
    time_s = time.time()

    for module in modules:
        module.pre_process()

    for module in modules:
        module.loop_pre_process()

    while True:
        for module in modules:
            if not module.inner_loop_pre_process():
                break
            if not module.inner_loop_main_process():
                break
            if not module.inner_loop_post_process():
                break
            if not module.check_error():
                break
            module.loop_count += 1
        else:
            time.sleep(sleep_time)
            continue
        break

    for module in modules:
        module.loop_post_process()

    for module in modules:
        module.post_process()

    report = CreationReaport(options)
    report.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()

    config_name = pathlib.Path(options['config']).name
    shutil.copy(pathlib.Path(options['config']), dst / config_name)

    files = fs.get_file_result_hp(dst)
    best, best_file = pt.get_best_parameter(files, goal, dict_lock)

    logger.info(f"Best result    : {best_file}")
    logger.info(f"               : {best}")
    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
