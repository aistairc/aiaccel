from pathlib import Path
import aiaccel
import logging
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.trialid import TrialId


class AbstractEvaluator(object):
    """An abstract class for MaximizeEvaluator and MinimizeEvaluator.

    """

    def __init__(self, options: dict) -> None:
        """Initial method for AbstractEvaluator.

        Args:
            config (ConfileWrapper): A configuration object.
        """
        self.options = options
        self.config_path = Path(self.options['config']).resolve()
        self.config = Config(str(self.config_path))
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
        best_trial_id_str = self.get_zero_padding_any_trial_id(best_trial_id)
        self.hp_result = self.storage.get_hp_dict(best_trial_id_str)

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

    def get_best_parameter(self) -> dict:
        """Get a best parameter in specified files.

        Args:
            files (List[Path]): A list of files to find a best.
            goal (str): Maximize or Minimize.
            dict_lock (Path): A directory to store lock files.

        Returns:
            Tuple[Union[float, None], Union[Path, None]]: A best result value and a
                file path. It returns None if a number of files is less than one.
        """
        best_trial_id, best_value = self.storage.get_best_trial(self.goal)
        return {
            'trial_id': best_trial_id,
            'value': best_value
        }
