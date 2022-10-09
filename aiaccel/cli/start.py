import pathlib
import shutil
import time
from argparse import ArgumentParser

import aiaccel
from aiaccel import parameter as pt
from aiaccel.config import Config
from aiaccel.util import filesystem as fs
from aiaccel.util import Terminal, Reporter
from aiaccel.workspace import Workspace
from aiaccel.optimizer.create import create_optimizer
from aiaccel.scheduler.create import create_scheduler
from aiaccel.master.create import create_master


def main() -> None:
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--clean', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if not pathlib.Path(args.config).exists():
        Terminal().print_error(f"config file: {args.config} doesn't exist.")
        return

    config = Config(args.config, warn=True, format_check=True)
    if config is None:
        Terminal().print_error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return

    workspace = Workspace(config.workspace.get())
    goal = config.goal.get()
    dict_lock = workspace.path / aiaccel.dict_lock

    if args.resume is None:
        if args.clean is True:
            print("Cleaning workspace")
            workspace.clean()
            print(f'Workspace directory {str(workspace.path)} is cleaned.')
        else:
            if workspace.exists():
                Terminal().print_error("workspace exists.")
                return

    workspace.create()
    if workspace.check_consists() is False:
        print("Creating workspace is Failed.")
        return

    print(f"Start {config.search_algorithm.get()} Optimization")
    print(f"config: {str(pathlib.Path(args.config).resolve())}")

    time_s = time.time()

    Master = create_master(args.config)
    Optimizer = create_optimizer(args.config)
    Scheduler = create_scheduler(args.config)
    modules = [Master(args), Scheduler(args), Optimizer(args)]

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

    reporter = Reporter(args.config)
    reporter.create()

    print("moving...")
    dst = workspace.move_completed_data()

    config_name = pathlib.Path(args.config).name
    shutil.copy(pathlib.Path(args.config), dst / config_name)

    files = fs.get_file_result_hp(dst)
    best, best_file = pt.get_best_parameter(files, goal, dict_lock)

    print(f"Best result    : {best_file}")
    print(f"               : {best}")
    print(f"Total time [s] : {round(time.time() - time_s)}")
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
