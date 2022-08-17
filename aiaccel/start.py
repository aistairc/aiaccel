import argparse
import pathlib
import shutil
import time
import multiprocessing
import csv
from typing import Any
from fasteners import InterProcessLock

import aiaccel
from aiaccel.master.abci_master import AbciMaster
from aiaccel.master.local_master import LocalMaster
from aiaccel.scheduler.abci_scheduler import AbciScheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.util import filesystem as fs
from aiaccel.easy_visualizer import EasyVisualizer
from aiaccel import parameter as pt
from aiaccel.util.filesystem import retry
from aiaccel.util.suffix import Suffix
from aiaccel.util.buffer import Buffer
from aiaccel.config import Config
from aiaccel.util.terminal import Terminal
from pathlib import Path
from importlib import import_module
from importlib.machinery import SourceFileLoader


class Arguments:
    def __new__(cls):
        p = argparse.ArgumentParser(allow_abbrev=False)
        p.add_argument('--config', '-c', type=str)
        p.add_argument('--resume', type=int, default=None)
        p.add_argument('--clean', nargs='?', const=True, default=False)
        p.add_argument('--nosave', nargs='?', const=True, default=False)
        p.add_argument('--dbg', nargs='?', const=True, default=False)
        p.add_argument('--graph', nargs='?', const=True, default=False)
        return vars(p.parse_args())


class Notification:
    """ Check the progress of the optimization and output it.

    Attributes:
        config (Config)         : Configuration object.
        options (dict)    : Command line options. (Argumens)
        resource (str)    : Rresource info. (local or ABCI)
        goal (str)        : Optimization goal. (maximize or minimize)
        trial_number (int): Max trial number.
        ws (pathlib.Path) : Path of 'workspace'.
        ws_hp (dict)      : Path of directories in 'hp' in 'workspace'.
        cplot (EasyVisualize) : Ascii graph plot.
        jobs_finished (list): Job list of finished.
        jobs_running (list) : Job list of running.
        jobs_ready (list)   : Job list of ready.
        buff (Buffer)    : Buffers.
        stats (list): Job status of state machine.
                        Refers 'job_thread.py' for detail.
        lock_file (dict) : Path of lock files.
        _error (bool)    : True if error occurs else false.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.options = Arguments()
        self.resource = self.config.resource_type.get().lower()
        self.goal = self.config.goal.get().lower()
        self.trial_number = self.config.trial_number.get()
        self.ws = pathlib.Path(self.config.workspace.get()).resolve()
        self.ws_hp = {
            'ready': self.ws / 'hp' / 'ready',
            'running': self.ws / 'hp' / 'running',
            'finished': self.ws / 'hp' / 'finished'
        }
        self.cplt = EasyVisualizer()
        self.jobs_finished = []
        self.jobs_running = []
        self.jobs_ready = []
        self.buff = Buffer(
            [
                'num_ready',  # num of hp-file in ready
                'num_running',  # num of hp-file in running
                'num_finished',  # num of hp-file in finished
                'ready_hash_name',  # hashname list in ready
                'running_hash_name',  # hashname list in running
                'finished_hash_name',  # hashname list in finished
                'results',  # result yaml data
                'objective',
                'best_value',
                'best_value_update_index',  # index of 'best_value' when
                                            # there is a difference
                'qstat',  # result of abci command 'qsub' as dictionaly
                'process_time',
            ]
        )
        # Only the two most recent data are needed for the buffer to
        # detect a difference.
        # The older data is not needed and will be deleted.
        buff_names = [
            'num_ready',
            'num_running',
            'num_finished',
            'finished_hash_name'
        ]
        for buff_name in buff_names:
            self.buff.d[buff_name].set_max_len(2)

        self.stats = []
        self.lock_file = {
            'result_txt': str(self.ws / 'lock' / 'result_txt'),
            'finished': str(self.ws / 'lock' / 'finished'),
        }
        self.start_time = 0.0
        self.jobs_aht = 0.0  # AHT/ Average Handling Time
        self.estimated_time = 0.0
        self._error = False

    def check_progress(self) -> None:
        """
        * Get list of workspase/hp/ready/*.hp
        * Get list of workspase/hp/running/*.hp
        * Get list of workspase/hp/finished/*.hp
        """
        self.jobs_ready = list(self.ws_hp['ready'].glob("*.hp"))
        self.jobs_running = list(self.ws_hp['running'].glob("*.hp"))
        self.jobs_finished = list(self.ws_hp['finished'].glob("*.hp"))
        # Store to buffer array
        self.buff.d['num_ready'].Add(len(self.jobs_ready))
        self.buff.d['num_running'].Add(len(self.jobs_running))
        self.buff.d['num_finished'].Add(len(self.jobs_finished))

        if (
            self.buff.d['num_ready'].has_difference() or
            self.buff.d['num_running'].has_difference() or
            self.buff.d['num_finished'].has_difference()
        ):
            if self.buff.d['finished_hash_name'].Len == 0:
                self.buff.d['finished_hash_name'].Add([])
            self.buff.d['finished_hash_name'].Add(self.jobs_finished)

    @property
    def progress(self) -> str:
        """ Get trial progress as {finished}/{max_trial}.

        Returns:
            progress (str): trial progress.
        """
        return "{}/{}".format(len(self.jobs_finished), self.trial_number)

    @property
    def status(self) -> str:
        """ Get number of hp-files(yaml) in ready, running,
            and finished directories in hp in workspace.

        Returns:
            str: Number of jobs that are ready, running, or finished.
        """
        return (
            "ready: {}, running: {}, finished: {}"
            .format(
                self.buff.d['num_ready'].Now,
                self.buff.d['num_running'].Now,
                self.buff.d['num_finished'].Now
            )
        )

    def diff_hp_file_list(self) -> list:
        """ Get the difference between the last sampling and the
            current sampling of the files in the finished directory.

        Returns:
            list[str]
        """
        if self.buff.d['num_finished'].has_difference():
            if self.buff.d['finished_hash_name'].Len >= 2:
                return self.buff.d['finished_hash_name'].iterable_diff()
            else:
                return []
        else:
            return []

    def get_objective(self) -> None:
        """ Get objective value(user function's retuen value)
            from hp-file in finished job.

            The objective value will be appended to buffer.

            If the user function encounters any errors,
            it will record them in the hp-file and
            this function will print them out.

        Returns:
            None
        """
        diffs = self.diff_hp_file_list()
        if diffs == []:
            return

        for file in diffs:
            with InterProcessLock(self.lock_file['finished']):
                fp = self.ws_hp['finished'] / file
                contents = fs.load_yaml(fp)
                if contents is None:
                    Terminal().print_error(
                        "Failed hp-file read: {}\n"
                        "contents:\n{}"
                        .format(str(fp), contents)
                    )
                elif 'error' in contents.keys():
                    Terminal().print_error("{}".format(contents['error']))
                    self.set_error()
                else:
                    self.buff.d['objective'].Add(contents['result'])
                    self.buff.d['results'].Add(contents)

    @property
    def error(self) -> bool:
        """ Get the error flag.
            True: There is an error.
            False: There is no error..

        Returns:
            bool: [True: Has error, False: No error]
        """
        return self._error

    def set_error(self) -> None:
        """ Set the error flag to True.

        Returns:
            None
        """
        self._error = True

    def clear_error(self) -> None:
        """ Set the error flag to False.

        Returns:
            None
        """
        self._error = False

    def update_best_value(self) -> None:
        """ Update the best objective value.

        Returns:
            None
        """
        sorted = []
        mx = float('-inf')
        mn = float('inf')

        for value in self.buff.d['objective'].Data:
            if self.goal.lower() == 'minimize':
                if value < mn:
                    mn = value
                    sorted.append(value)
                else:
                    sorted.append(mn)

            elif self.goal.lower() == 'maximize':
                if value > mx:
                    mx = value
                    sorted.append(value)
                else:
                    sorted.append(mx)

        num_diffs = (len(sorted) - self.buff.d['best_value'].Len)
        if num_diffs > 0:
            diffs = sorted[len(sorted) - num_diffs:len(sorted)]
            for value in diffs:
                self.buff.d['best_value'].Add(value)
            self.buff.d['best_value_update_index'].Add(
                self.buff.d['best_value'].Len - 1
            )

    @property
    def best_value_has_difference(self) -> bool:
        """ Check best value has difference or not.

        Returns:
            bool: true if best value has change else false.
        """
        if self.buff.d['best_value_update_index'].Len >= 2:
            idx_pre = self.buff.d['best_value_update_index'].Pre
            idx_now = self.buff.d['best_value_update_index'].Now
            if self.buff.d['best_value'].point_difference(
                idx_pre=idx_pre,
                idx_now=idx_now
            ):
                return True
        return False

    def _update(self) -> None:
        self.check_progress()
        self.get_objective()
        self.update_best_value()

    def make_message(self) -> list:
        """ Make print message.

        Returns:
            list[str]: The text will be displayed on the terminal.
        """
        self._update()
        mess = []

        # progress
        if (
            self.buff.d['num_ready'].has_difference() or
            self.buff.d['num_running'].has_difference() or
            self.buff.d['num_finished'].has_difference()
        ):
            mess.append(
                "{}progress {}{}status {}"
                .format(
                    "\t", self.progress,
                    "\t", self.status,
                )
            )

        if (
            self.buff.d['num_finished'].has_difference() and
            self.best_value_has_difference
        ):
            idx_pre = self.buff.d['best_value_update_index'].Pre
            idx_now = self.buff.d['best_value_update_index'].Now
            pre = self.buff.d['best_value'].Value(idx_pre)
            now = self.buff.d['best_value'].Value(idx_now)
            mess.append("update best value: {} -> {}".format(pre, now))
        return mess

    def plot(self) -> None:
        """ Plot ascii graphs.

        Returns:
            None

        Note:
            Plot data are objectives and best values.
        """
        if self.options['graph'] is False:
            return

        self.cplt.set_colors(
            [
                "red",
                "green"
            ]
        )
        self.cplt.caption(
            [
                "objective",
                "best value"
            ]
        )
        self.cplt.line_plot(
            [
                self.buff.d['objective'].Data,
                self.buff.d['best_value'].Data
            ]
        )

    def display(self) -> None:
        """ Displays progress message and ascii graph.

        Returns:
            None
        """
        mess = self.make_message()
        if mess is None:
            return
        if mess == "":
            return

        if self.options['dbg'] is False:
            for m in mess:
                print(m)

        # graph
        if (
            self.buff.d['num_finished'].has_difference() and
            self.best_value_has_difference
        ):
            self.plot()

    @property
    def curr_trial_number(self) -> int:
        """ Get current trial number.

        Returns:
            int: Current trial number.

        Note:
            The reality is number of hp-file in finished.
        """
        return len(list(self.ws_hp['finished'].glob("*.hp")))

    def write_to_txt(self) -> None:
        """ Create report from any buffers data when all finished.

        Returns:
            None
        """
        self._update()

        fp = self.ws / 'results.csv'
        data = []
        header = []

        if self.buff.d['results'].Len == 0:
            print('No buffer.')
            return

        # Write header
        latest = self.buff.d['results'].Value(-1)
        header.append('hashname')
        for param in latest['parameters']:
            header.append(param['parameter_name'])
        header.append('objective')

        with InterProcessLock(self.lock_file['result_txt']):
            with open(fp, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(header)

        # Write result data
        for contents in self.buff.d['results'].Data:
            row = []
            row.append(str(contents['hashname']))
            for param in contents['parameters']:
                row.append(param['value'])
            row.append(contents['result'])
            data.append(row)

        with InterProcessLock(self.lock_file['result_txt']):
            with open(fp, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(data)


class CreationMaster:
    """ Create a master class
        by selecting localmaster or abcimaster.

    Example:
        master = CreationMaster(config_path)
    """

    def __init__(self, config_path: str) -> None:
        config = Config(config_path)
        self.resource = config.resource_type.get()
        config = None

    def __call__(self) -> Any:
        if self.resource.lower() == "local":
            self.master = LocalMaster

        elif self.resource.lower() == "abci":
            self.master = AbciMaster

        else:
            self.master = None

        return self.master


class CreationScheduler:
    """ Create a scheduler class
        by selecting localscheduler or abcischeduler.

    Example:
        scheduler = CreationScheduler(config_path)
    """

    def __init__(self, config_path: str) -> None:
        config = Config(config_path)
        self.resource = config.resource_type.get()
        config = None

    def __call__(self) -> Any:
        if self.resource.lower() == "local":
            self.scheduler = LocalScheduler

        elif self.resource.lower() == "abci":
            self.scheduler = AbciScheduler

        else:
            self.scheduler = None

        return self.scheduler


class CreationOptimizer:
    """ Create an optimizer class by selecting
        search_algirithm_grid or
        search_algorithm_nelder_mead or
        search_algorithm_random or
        search_algorithm_sobol or
        search_algorithm_tpe.

    Example:
        optimizer = CreationOptimizer(config_path)
    """

    def __init__(self, config_path: str) -> None:
        config = Config(config_path)
        self.optimizer = self.import_and_getattr(config.search_algorithm.get())

    def __call__(self) -> Any:
        return self.optimizer

    def get_pyfile_path(self, module_name: str) -> str:
        path = (
            module_name
            .replace(".", "/")
            .replace("//", "../")
            .replace(" ", "")
        ) + ".py"
        return path

    def import_and_getattr(self, name: str) -> Any:
        module_name, attr_name = name.rsplit(".", 1)
        pyfile = self.get_pyfile_path(module_name)

        if Path(pyfile).exists():
            # Import external modules not included in the aiaccel package.
            module = SourceFileLoader('MyOptimizer', pyfile).load_module()
        else:
            # It is assumed to be imported from aiaccel.optimizer.*.
            module = import_module(module_name)
        return getattr(module, attr_name)


class Master(CreationMaster(Arguments()['config'])()):
    """ The Master class.

    The mainly Master role is to start the Optimizer and Scheduler,
    and to monitor their activity. (Life and death monitoring)
    """

    def __init__(self) -> None:
        options = Arguments()
        options['process_name'] = 'master'
        super().__init__(options)
        barrier = multiprocessing.Barrier(3)
        self.set_barrier(barrier)
        self.o = Optimizer()
        self.o.set_barrier(barrier)
        self.s = Scheduler()
        self.s.set_barrier(barrier)
        self.worker_o = multiprocessing.Process(target=self.o.start)
        self.worker_s = multiprocessing.Process(target=self.s.start)
        self.notification = Notification(self.config)

    def start_optimizer(self) -> None:
        """ Start the Optimizer process.

        Returns:
            None
        """
        self.worker_o.daemon = True
        self.worker_o.start()

    def start_scheduler(self) -> None:
        """ Start the Scheduler process.

        Returns:
            None
        """
        self.worker_s.daemon = True
        self.worker_s.start()

    def other_process_is_alive(self) -> bool:
        if (
            not self.worker_o.is_alive() or
            not self.worker_s.is_alive()
        ):
            return False
        return True

    def print_dict_state(self) -> None:
        """ Display number of hp-files in ready, running, finished directory.

        Returns:
            None
        """
        super().print_dict_state()
        self.notification.display()

    def plot(self) -> None:
        """ Plot ascii graph.

        It is assumed that it will be executed last.

        Returns:
            None
        """
        self.notification.plot()

    def check_error(self) -> bool:
        if self.notification.error is True:
            self.stop()
            return False
        return True


class Scheduler(CreationScheduler(Arguments()['config'])()):
    """ The Scheduler class

    The Scheduler role is to submit any job.
    """

    def __init__(self) -> None:
        options = Arguments()
        options['process_name'] = 'scheduler'
        super().__init__(options)


class Optimizer(CreationOptimizer(Arguments()['config'])()):
    """ The Optimizer class

    The Optimizer role is to calculate hyperparameter value.
    """

    def __init__(self) -> None:
        options = Arguments()
        options['process_name'] = 'optimizer'
        super().__init__(options)


@retry(_MAX_NUM=300, _DELAY=1.0)
def clean_workspace(workspace: pathlib.Path) -> None:
    """ Delete a workspace.

    It is assumed to be the first one to be executed.
    """
    if not workspace.exists():
        return
    shutil.rmtree(workspace)
    return


def delete_alive(ws: pathlib.Path) -> None:
    """ Delete alive-files in alive directory in workspace.

    Stops the optimization process when executed.
    """
    alive = ws.resolve() / 'alive'
    files = alive.glob("*.*")
    for file in files:
        file.unlink()


def move_completed_data(ws: pathlib.Path) -> pathlib.Path:
    """ Move workspace to under of results directory when finished.
    """
    # ws: workspace
    results = pathlib.Path("./results")
    dst = results / Suffix.date()
    if not results.exists():
        results.mkdir()

    if dst.exists():
        return

    shutil.copytree(ws, dst)
    return dst


def main(options: dict = Arguments()) -> None:
    if options['config'] == "" or options['config'] is None:
        Terminal().print_error(
            "The configuration file is not specified. "
            "Please specify it with the command line argument "
            "'--config' or '-c'."
        )
        return

    if not pathlib.Path(options['config']).exists():
        Terminal().print_error(
            "config file: {} doesn't exist.".format(options['config'])
        )
        return

    config = Config(options['config'], warn=True, format_check=True)
    if config is None:
        Terminal().print_error(
            'Invalid workspace: {} or config: {}'
            .format(options['workspace'], options['config'])
        )
        return

    workspace = pathlib.Path(config.workspace.get()).resolve()
    goal = config.goal.get()
    dict_lock = workspace / aiaccel.dict_lock

    if options['resume'] is None:
        if options['clean'] is True:
            print("Cleaning workspace")
            clean_workspace(workspace)

            time.sleep(config.sleep_time_master.get())
            # Waiting for modification of launched processes

            print(
                'Workspace directory {} is cleaned.'
                .format(workspace)
            )
        else:
            if workspace.exists():
                Terminal().print_error("workspace exists.")
                return
    else:
        delete_alive(workspace)

    print("Start {} Optimization".format(config.search_algorithm.get()))
    print("config: {}".format(str(pathlib.Path(options['config']).resolve())))

    time_s = time.time()
    M = Master()
    M.start()

    #
    # Wait for finish
    #

    if options['graph'] is True:
        print("Final")
        M.plot()
    M.notification.write_to_txt()

    print("moving...")
    dst = move_completed_data(workspace)

    config_name = pathlib.Path(options['config']).name
    shutil.copy(pathlib.Path(options['config']), dst / config_name)

    files = fs.get_file_hp_finished(dst)
    best, best_file = pt.get_best_parameter(files, goal, dict_lock)
    print("Best result    : {}".format(best_file))
    print("               : {}".format(best))
    print("Total time [s] : {}".format(round(time.time() - time_s)))
    print("Done.")
    return


if __name__ == "__main__":
    main()
