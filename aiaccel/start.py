import pathlib
import shutil
import time
import aiaccel
from aiaccel import parameter as pt
from aiaccel.config import Config
from aiaccel.util import filesystem as fs
from aiaccel.util.terminal import Terminal
from aiaccel.util.report import CreationReaport
from aiaccel.util.trialid import TrialId
from aiaccel.storage.storage import Storage
from aiaccel.argument import Arguments
from aiaccel.master.create import create_master
from aiaccel.workspace import Workspace


def main(options: dict = Arguments()) -> None:
    if options['config'] == "" or options['config'] is None:
        Terminal().print_error(
            "The configuration file is not specified. "
            "Please specify it with the command line argument "
            "'--config' or '-c'."
        )
        return

    if not pathlib.Path(options['config']).exists():
        Terminal().print_error(f"config file: {options['config']} doesn't exist.")
        return

    config = Config(options['config'], warn=True, format_check=True)
    if config is None:
        Terminal().print_error(f"Invalid workspace: {options['workspace']} or config: {options['config']}")
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
                Terminal().print_error("workspace exists.")
                return

    workspace.create()
    if workspace.check_consists() is False:
        print("Creating workspace is Failed.")
        return

    storage = Storage(
        workspace.path,
        fsmode=options['fs'],
        config_path=config.config_path
    )
    storage.alive.init_alive()

    if options['resume'] is not None:
        resume_trial_id = int(options['resume'])
        storage.rollback_to_ready(resume_trial_id)
        storage.delete_trial_data_after_this(resume_trial_id)
        trial_id = TrialId(options['config'])
        trial_id.initial(num=resume_trial_id - 1)
    storage = None

    print(f"Start {config.search_algorithm.get()} Optimization")
    print(f"config: {str(pathlib.Path(options['config']).resolve())}")

    time_s = time.time()
    M = create_master(options)

    M.start()

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
