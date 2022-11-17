import logging
from pathlib import Path

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.trialid import TrialId

from typing import Union


class AbstractEvaluator(object):
    """An abstract class for MaximizeEvaluator and MinimizeEvaluator.

    """

    def __init__(self, config_path: Union[Path, str]) -> None:
        """Initial method for AbstractEvaluator.

        Args:
            config (ConfileWrapper): A configuration object.
        """

        # === Load config file===
        self.config_path = config_path
        if type(self.config_path) == str:
            self.config_path = Path(self.config_path)
        self.config_path = self.config_path.resolve()

        self.config = Config(self.config_path)
        self.ws = Path(self.config.workspace.get()).resolve()
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.hp_result = None
        self.storage = Storage(self.ws)
        self.goal = self.config.goal.get()
        self.trial_id = TrialId(str(self.config_path))

    def get_zero_padding_any_trial_id(self, trial_id: int):
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
        logger = logging.getLogger('root.evaluator')
        logger.info('Best hyperparameter is followings:')
        logger.info(self.hp_result)

    def save(self) -> None:
        """Save current results to a file.

        Returns:
            None
        """
        path = self.ws / aiaccel.dict_result / aiaccel.file_final_result
        create_yaml(path, self.hp_result, self.dict_lock)
