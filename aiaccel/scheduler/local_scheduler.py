import re
import subprocess
import threading
import importlib

from typing import Union
from pathlib import Path

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.aiaccel import Run
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.process import ps2joblist
from aiaccel.wrapper_tools import create_runner_command
from aiaccel.util.time_tools import get_time_now
from aiaccel import dict_result


class LocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer.

    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)

        self.run = None
        self.user_func = None

        if self.is_python_local_mode():
            self.user_func = self.get_callable_object(
                self.config.python_file.get(),
                self.config.function.get()
            )
            self.run = Run(self.config_path)

    def is_python_local_mode(self):
        return self.config.resource_type.get().lower() == 'python_local'

    def get_stats(self) -> None:
        """Get a current status and update.

        Returns:
            None
        """
        super().get_stats()

        res = ps2joblist()
        command = self.config.job_command.get()
        self.stats = []
        trial_id_list = [job['trial_id'] for job in self.jobs]

        for r in res:
            if command in r['name']:
                trial_id = int(self.parse_trial_id(r['name']))

                if trial_id in trial_id_list:
                    self.stats.append(r)
                else:
                    self.logger.warning(f'**** Unknown process: {r}')

    def parse_trial_id(self, command: str) -> Union[None, str]:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str: An unique name.
        """
        args = re.split(' +', command)
        # args:
        # ['2', 'python', 'user.py', '--trial_id', '2',
        # '--config', 'config.yaml',
        #  '--x1=3.65996970905703', '--x2=2.99329242098518']
        #
        trial_id_index = args.index('--trial_id')
        index_offset = 1

        if trial_id_index is None:
            return None

        return args[trial_id_index + index_offset]

    def get_callable_object(self, file_path: Union[str, Path], attr_name: str) -> callable:
        """ Loads the specified module from the specified python program.

        Args:
            file_path (str, pathlib.Path): A user program file path.(python file only)
            attr_name (str): A name of objective function in user program.

        Returns:
            callable
        """
        spec = importlib.util.spec_from_file_location("user_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, attr_name)

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """

        trial_ids = self.storage.trial.get_ready()
        if trial_ids is None or len(trial_ids) == 0:
            return True

        for trial_id in trial_ids:
            self._serialize(trial_id)
            if self.num_node > 1:
                th = threading.Thread(target=self.execute, args=(trial_id,))
                th.start()
            else:
                self.execute(trial_id)

        return True

    def execute(self, trial_id: int) -> None:
        """ Execute in subprocess or directly from a loaded callable object.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """
        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='running')

        if self.is_python_local_mode():
            self.execute_callable(trial_id)
        else:
            self.execute_in_subprocess(trial_id)

        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')
        self.create_result_file(trial_id)

        return

    def execute_in_subprocess(self, trial_id: int) -> None:
        """ Generates and executes commands to run user programs.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """
        runner_command = create_runner_command(
            self.config.job_command.get(),
            self.storage.get_hp_dict(trial_id),
            trial_id,
            self.config_path
        )
        subprocess.run(runner_command)

        return

    def execute_callable(self, trial_id: int) -> None:
        """ Executes the loaded callable object.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """
        start_time = get_time_now()
        xs, y, err = self.run.execute(self.user_func, trial_id, y_data_type=None)
        end_time = get_time_now()
        self.run.report(trial_id, xs, y, err, start_time, end_time)

        return

    def create_result_file(self, trial_id: int) -> None:
        """ Save the results in yaml format.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """

        file_hp_count_fmt = f'%0{self.config.name_length.get()}d'
        file_name = file_hp_count_fmt % trial_id + '.hp'

        content = self.storage.get_hp_dict(trial_id)
        result = self.storage.result.get_any_trial_objective(trial_id=trial_id)
        error = self.storage.error.get_any_trial_error(trial_id=trial_id)

        content['result'] = str(result)

        if error is not None:
            content['error'] = error

        result_file_path = self.ws / dict_result / file_name
        create_yaml(result_file_path, content)

    def __getstate__(self):
        obj = super().__getstate__()
        del obj['run']
        del obj['user_func']
        return obj
