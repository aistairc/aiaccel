from __future__ import annotations

import re

from aiaccel.manager.abstract_manager import AbstractManager
from aiaccel.manager.job.model.abci_model import AbciModel


class AbciManager(AbstractManager):
    """A manager class running on ABCI environment."""

    def parse_trial_id(self, command: str) -> str | None:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str | None: An unique name.
        """
        self.logger.debug(f"command: {command}")
        full = re.compile(r"run_\d{1,65535}.sh")
        numbers = re.compile(r"\d{1,65535}")
        if full.search(command) is None:
            return None
        if match := numbers.search(command):
            return match.group()
        else:
            return None

    def create_model(self) -> AbciModel:
        """Creates model object of state machine.

        Returns:
            AbciModel: Model object.
        """
        return AbciModel()
