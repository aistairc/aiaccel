from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aiaccel.common import dict_lock, dict_verification, extension_verification
from aiaccel.config import Config
from aiaccel.storage import Storage
from aiaccel.util import create_yaml


class AbstractVerifier(ABC):
    """An abstract class of verifier.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.
        config (Config): Config object
        ws (Path): Path to the workspace.
        dict_lock (Path): Path to "lock", i.e. `ws`/lock.
        is_verified (bool): Whether verified or not.
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
        self.ws = Path(self.config.workspace.get()).resolve()
        self.dict_lock = self.ws / dict_lock
        self.is_verified = self.config.is_verified.get()
        self.condition = self.config.condition.get()
        self.verification_result = copy.copy(self.condition)
        self.storage = Storage(self.ws)

    @abstractmethod
    def verify(self) -> None: ...

    def load_verification_config(self) -> None:
        """Load configurations about verification.
        """
        self.is_verified = self.config.is_verified.get()
        self.condition = self.config.condition.get()
        self.verification_result = copy.copy(self.condition)

    def print(self) -> None:
        """Print current verifications result.
        """
        if not self.is_verified:
            return

        logger = logging.getLogger('root.master.verifier')
        logger.info('Current verification is followings:')
        logger.info(f'{self.verification_result}')

    def save(self, name: str | int) -> None:
        """Save current verifications result to a file.

        Args:
            name (int):
        """
        if not self.is_verified:
            return

        path = self.ws / dict_verification / f'{name}.{extension_verification}'
        create_yaml(path, self.verification_result, self.dict_lock)
        logger = logging.getLogger('root.master.verifier')
        logger.info(f'Save verifiation file: {name}.{extension_verification}')
