import logging
from pathlib import Path

from omegaconf.dictconfig import DictConfig

from aiaccel import dict_lock
from aiaccel import dict_result
from aiaccel import file_final_result
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.trialid import TrialId


class AbstractEvaluator(object):
    """An abstract class for MaximizeEvaluator and MinimizeEvaluator.

    """

    def __init__(self, config: DictConfig) -> None:
        """Initial method for AbstractEvaluator.

        Args:
            config (ConfileWrapper): A configuration object.
        """
        self.config = config
        self.config_path = Path(self.config.config_path).resolve()
        self.ws = Path(self.config.generic.workspace).resolve()
        self.dict_lock = self.ws / dict_lock
        self.hp_result = None
        self.storage = Storage(self.ws)
        self.goal = self.config.optimize.goal
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
        logger = logging.getLogger('root.master.evaluator')
        logger.info('Best hyperparameter is followings:')
        logger.info(self.hp_result)

    def save(self) -> None:
        """Save current results to a file.

        Returns:
            None
        """
        path = self.ws / dict_result / file_final_result
        create_yaml(path, self.hp_result, self.dict_lock)
