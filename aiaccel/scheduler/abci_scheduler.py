from __future__ import annotations

import re
import subprocess

from aiaccel.abci.qstat import parse_qstat
from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.scheduler.job.model.abci_model import AbciModel


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
        stdout_data, _ = p.communicate()
        stats = stdout_data.decode('utf-8')

        if len(stats) < 1:
            return

        self.stats = parse_qstat(self.config, stats)

        for stat in self.stats:
            self.logger.info(
                f'stat job-ID: {stat["job-ID"]}, '
                f'name: {stat["name"]}, '
                f'state: {stat["state"]}'
            )

    def parse_trial_id(self, command: str) -> str | None:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str | None: An unique name.
        """
        self.logger.debug(f"command: {command}")
        full = re.compile(r'run_\d{1,65535}.sh')
        numbers = re.compile(r'\d{1,65535}')
        if full.search(command) is None:
            return None
        return numbers.search(command).group()

    def create_model(self) -> AbciModel:
        """Creates model object of state machine.

        Returns:
            AbciModel: Model object.
        """
        return AbciModel()
