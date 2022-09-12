import re
import subprocess
from typing import Union
from aiaccel.abci.qstat import parse_qstat
from aiaccel.scheduler.abstract import AbstractScheduler


class AbciScheduler(AbstractScheduler):
    """A scheduler class running on ABCI environment.
    """

    def get_stats(self) -> None:
        """Get a current status and update.

        Returns:
            None
        """
        super().get_stats()

        commands = 'qstat -xml'
        p = subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True)
        stats = p.communicate()[0]
        stats = stats.decode('utf-8')

        if len(stats) < 1:
            return

        self.stats = parse_qstat(self.config, stats)

        for stat in self.stats:
            self.logger.info(
                f'stat job-ID: {stat["job-ID"]}, '
                f'name: {stat["name"]}, '
                f'state: {stat["state"]}'
            )

    def parse_trial_id(self, command: str) -> Union[None, str]:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str: An unique name.
        """
        self.logger.debug(f"command: {command}")
        full = re.compile(r'run_\d{1,65535}.sh')
        numbers = re.compile(r'\d{1,65535}')
        if full.search(command) is None:
            return None
        return numbers.search(command).group()
