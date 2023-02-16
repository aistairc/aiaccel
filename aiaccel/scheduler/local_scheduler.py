from __future__ import annotations

import re

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.util.process import ps2joblist
from aiaccel.scheduler.job.model.local_model import LocalModel


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
        trial_id_list = [job.trial_id for job in self.jobs]

        for r in res:
            if command in r['name']:
                trial_id = int(self.parse_trial_id(r['name']))

                if trial_id in trial_id_list:
                    self.stats.append(r)
                else:
                    self.logger.warning(f'**** Unknown process: {r}')

    def parse_trial_id(self, command: str) -> str | None:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str | None: An unique name.
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

    def create_model(self) -> LocalModel:
        """Creates model object of state machine.

        Returns:
            LocalModel: Model object.
        """
        return LocalModel()
