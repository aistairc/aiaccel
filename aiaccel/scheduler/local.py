import re
from typing import Union
from aiaccel.scheduler.abstract import AbstractScheduler
from aiaccel.util.process import ps2joblist


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
                    self.logger.warning('**** Unknown process: {}'.format(r))

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
