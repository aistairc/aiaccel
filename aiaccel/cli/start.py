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
    if options['config'] == "" or options['config'] is None:
        print(
            "The configuration file is not specified. "
            "Please specify it with the command line argument "
            "'--config' or '-c'."
        )
        return

    if not pathlib.Path(options['config']).exists():
        print(f"config file: {options['config']} doesn't exist.")
        return

    config = Config(options['config'], warn=True, format_check=True)
    if config is None:
        print(f"Invalid workspace: {options['workspace']} or config: {options['config']}")
        return

    workspace = Workspace(config.workspace.get())
    goal = config.goal.get()
    dict_lock = workspace.path / aiaccel.dict_lock

    if options['resume'] is None:
        if options['clean'] is True:
            print("Cleaning workspace")
            workspace.clean()
            print(f'Workspace directory {str(workspace.path)} is cleaned.')
        else:
            if workspace.exists():
                print("workspace exists.")
                return

    workspace.create()
    if workspace.check_consists() is False:
        print("Creating workspace is Failed.")
        return

    print(f"Start {config.search_algorithm.get()} Optimization")
    print(f"config: {str(pathlib.Path(options['config']).resolve())}")

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

    print("moving...")
    dst = workspace.move_completed_data()

    config_name = pathlib.Path(options['config']).name
    shutil.copy(pathlib.Path(options['config']), dst / config_name)

    files = fs.get_file_result_hp(dst)
    best, best_file = pt.get_best_parameter(files, goal, dict_lock)

    print(f"Best result    : {best_file}")
    print(f"               : {best}")
    print(f"Total time [s] : {round(time.time() - time_s)}")
    print("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
