import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger
import re

import aiaccel
from aiaccel import parameter as pt
from aiaccel.config import Config
from aiaccel.master.create import create_master
from aiaccel.optimizer.create import create_optimizer
from aiaccel.scheduler.create import create_scheduler
from aiaccel.util import filesystem as fs
from aiaccel.util.report import CreationReport
from aiaccel.workspace import Workspace
import subprocess

parser = ArgumentParser()
parser.add_argument('--config', '-c', type=str, default="config.yml")
parser.add_argument('--resume', type=int, default=None)
parser.add_argument('--clean', nargs='?', const=True, default=False)
args = parser.parse_args()

import threading

logger = getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.addHandler(StreamHandler())

from aiaccel.wrapper_tools import create_runner_command
# from aiaccel.util.process import OutputHandler, exec_runner, kill_process


def local(config: Config):
    command = config.job_command.get()
    commands = re.split(' +', command)
    commands.append('--config')
    commands.append(str(config.config_path))
    if args.resume is not None:
        commands.append('--resume')
        commands.append(str(args.resume))
    subprocess.run(commands)

def abci(config: Config):
    Master = create_master(config.config_path)
    Optimizer = create_optimizer(config.config_path)
    Scheduler = create_scheduler(config.config_path)
    modules = [Master(vars(args)), Scheduler(vars(args)), Optimizer(vars(args))]

    sleep_time = config.sleep_time.get()

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


def main() -> None:  # pragma: no cover

    config = Config(args.config, warn=True, format_check=True)
    if config is None:
        logger.error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return

    workspace = Workspace(config.workspace.get())
    goal = config.goal.get()
    dict_lock = workspace.path / aiaccel.dict_lock
    resource_type = config.resource_type.get()

    if args.resume is None:
        if args.clean is True:
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

    logger.info(f"config: {str(pathlib.Path(args.config).resolve())}")

    time_s = time.time()

    if resource_type.lower() == 'abci':
        abci(config)
    elif resource_type.lower() == 'local':
        local(config)

    report = CreationReport(args.config)
    report.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()

    config_name = pathlib.Path(args.config).name
    shutil.copy(pathlib.Path(args.config), dst / config_name)

    files = fs.get_file_result_hp(dst)
    best, best_file = pt.get_best_parameter(files, goal, dict_lock)

    logger.info(f"Best result    : {best_file}")
    logger.info(f"               : {best}")
    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
