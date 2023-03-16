from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.trialid import TrialId


class AbstractEvaluator(object):
    """An abstract class for MaximizeEvaluator and MinimizeEvaluator.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.
        config_path (Path): Path to the configuration file.
        config (Config): Config object.
        ws (Path): Path to the workspace.
        dict_lock (Path): Path to "lock', i.e. `ws`/lock.
        hp_result (dict): A dict object of the best optimized result.
        storage (Storage): Storage object.
        goal (str): Goal of optimization ('minimize' or 'maximize').
        trial_id (TrialId): TrialId object.

    """

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self.config_path = Path(self.options['config']).resolve()
        self.config = Config(str(self.config_path))
        self.ws = Path(self.config.workspace.get()).resolve()
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.hp_result: dict[str, Any] | None = None
        self.storage = Storage(self.ws)
        self.goal = self.config.goal.get()
        self.trial_id = TrialId(self.config)

    def get_zero_padding_any_trial_id(self, trial_id: int) -> str:
        """Returns string of trial id padded by zeros.

        Args:
            trial_id (int): Trial id.

        Returns:
            str: Trial id padded by zeros.
        """
        return self.trial_id.zero_padding_any_trial_id(trial_id)

    def evaluate(self) -> None:
        """Run an evaluation.

        Returns:
            None
        """
        best_trial_id, _ = self.storage.get_best_trial(self.goal)
        self.hp_result = self.storage.get_hp_dict(best_trial_id)

    def print(self) -> None:
        """Print current results.

        Returns:
            None
        """
        logger = logging.getLogger('root.master.evaluator')
        logger.info('Best hyperparameter is followings:')
        logger.info(self.hp_result)

    def save(self) -> None:
        """Save current results to a file.

        Returns:
            None
        """
        path = self.ws / aiaccel.dict_result / aiaccel.file_final_result
        create_yaml(path, self.hp_result, self.dict_lock)
