from __future__ import annotations

import copy
import logging
from typing import Any

from aiaccel import extension_verification
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import create_yaml
from aiaccel.workspace import Workspace


class AbstractVerification(object):
    """An abstract class of verification.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.
        config (Config): Config object
        workspace (Workspace): Path to the workspace.
        is_verified (bool): Whether verified or not.
        finished_loop (int): The last loop number verified.
        condition (list[dict[str, int | float]]): A list of verification
            conditions. Each element dict has keys 'loop', 'minimum', and
            'maximum' with values as int, float, and float, respectively.
            The verification is `True` if the optimized result of which the
            trial id is the given 'loop' is greater than or equal to
            'minimum' as well as is less than or equal to the 'maximum'.
        verification_result (list[dict]): Deepcopy object of condition.
        storage (Storage): Storage object.
    """

    def __init__(self, options: dict[str, Any]) -> None:
        # === Load config file===
        self.options = options
        self.config = Config(self.options['config'])
        self.workspace = Workspace(self.config.workspace.get())
        self.is_verified: bool = None
        self.finished_loop = None
        self.condition: Any = None
        self.verification_result: Any = None
        self.load_verification_config()
        self.storage = Storage(self.workspace.path)

    def verify(self) -> None:
        """Run a verification.

        The trigger to run a verification, is described in configuration file
        'verification' > 'conditions'.

        Returns:
            None
        """
        if not self.is_verified:
            return

        for i, c in enumerate(self.condition):
            if self.storage.get_num_finished() >= c['loop']:
                if (
                    self.finished_loop is None or
                    c['loop'] > self.finished_loop
                ):
                    self.make_verification(i, c['loop'])
                    self.finished_loop = c['loop']

    def make_verification(self, index: int, loop: int) -> None:
        """Run a verification and save the result.

        Args:
            index (int): An index of verifications.
            loop (int): A loop count of Master.

        Returns:
            None
        """
        best_trial = self.storage.get_best_trial_dict(self.config.goal.get().lower())

        if (
            best_trial['result'] < self.condition[index]['minimum'] or
            best_trial['result'] > self.condition[index]['maximum']
        ):
            self.verification_result[index]['passed'] = False
        else:
            self.verification_result[index]['passed'] = True
        self.save(loop)

    def load_verification_config(self) -> None:
        """Load configurations about verification.

        Returns:
            None
        """
        self.is_verified = self.config.is_verified.get()
        self.condition = self.config.condition.get()
        self.verification_result = copy.copy(self.condition)

    def print(self) -> None:
        """Print current verifications result.

        Returns:
            None
        """
        if not self.is_verified:
            return None

        logger = logging.getLogger('root.master.verification')
        logger.info('Current verification is followings:')
        logger.info(f'{self.verification_result}')

    def save(self, name: str | int) -> None:
        """Save current verifications result to a file.

        Args:
            name (int):

        Returns:
            None
        """
        if not self.is_verified:
            return None

        path = self.workspace.verification / f'{name}.{extension_verification}'
        create_yaml(path, self.verification_result, self.workspace.lock)
        logger = logging.getLogger('root.master.verification')
        logger.info(f'Save verifiation file: {name}.{extension_verification}')
