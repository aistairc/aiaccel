import re
import subprocess
import threading

from typing import Union

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.process import ps2joblist
from aiaccel.wrapper_tools import create_runner_command


class LocalScheduler(AbstractScheduler):
    """A scheduler class running on a local computer.

    """

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
        """ Generates and executes commands to run user programs.

        Args:
            trial_id (int): Any trial od

        Returns:
            None
        """
        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='running')

        runner_command = create_runner_command(
            self.config.job_command.get(),
            self.storage.get_hp_dict(trial_id),
            trial_id,
            self.config_path
        )
        subprocess.run(runner_command)

        self.storage.trial.set_any_trial_state(trial_id=trial_id, state='finished')
        self.create_result_file(trial_id)

        return
