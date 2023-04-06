from __future__ import annotations

import logging
from typing import Any

from omegaconf.dictconfig import DictConfig

from aiaccel.common import file_final_result
from aiaccel.storage import Storage
from aiaccel.util import TrialId, create_yaml
from aiaccel.workspace import Workspace


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
        hp_result (dict): A dict object of the best optimized result.
        storage (Storage): Storage object.
        goals (list[str]): Goal of optimization ('minimize' or 'maximize').
        trial_id (TrialId): TrialId object.

    """
    def __init__(self, config: DictConfig) -> None:
        """Initial method for AbstractEvaluator.

        Args:
            config (ConfileWrapper): A configuration object.
        """
        self.config = config
        self.workspace = Workspace(self.config.generic.workspace)
        self.hp_result: list[dict[str, Any]] | None = None
        self.storage = Storage(self.workspace.path)
        self.goals = [item.value for item in self.config.optimize.goal]
        self.trial_id = TrialId(self.config)

    def evaluate(self) -> None:
        """Run an evaluation.

        Returns:
            None
        """
        best_trial_ids, _ = self.storage.get_best_trial(self.goals)
        if best_trial_ids is None:
            return

        hp_results: list[dict[str, Any]] = []
        for best_trial_id in best_trial_ids:
            hp_results.append(self.storage.get_hp_dict(best_trial_id))
        self.hp_result = hp_results

    def print(self) -> None:
        """Print current results.

        Returns:
            None
        """
        logger = logging.getLogger('root.master.evaluator')
        if self.hp_result:
            logger.info('Best hyperparameter is followings:')
            logger.info(self.hp_result)
        else:
            logger.info('Evaluation not available (no results in storage.db).')

    def save(self) -> None:
        """Save current results to a file.

        Returns:
            None
        """
        path = self.workspace.result / file_final_result
        create_yaml(path, self.hp_result, self.workspace.lock)
