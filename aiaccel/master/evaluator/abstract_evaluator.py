from aiaccel.util.filesystem import create_yaml, load_yaml
from pathlib import Path
import aiaccel
import logging
from aiaccel.config import Config


class AbstractEvaluator(object):
    """An abstract class for MaximizeEvaluator and MinimizeEvaluator.

    """

    def __init__(self, config: Config) -> None:
        """Initial method for AbstractEvaluator.

        Args:
            config (ConfileWrapper): A configuration object.
        """
        self.config = config
        self.ws = Path(self.config.workspace.get()).resolve()
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.hp_result = None

    def evaluate(self) -> None:
        """Run an evaluation.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def print(self) -> None:
        """Print current results.

        Returns:
            None
        """
        logger = logging.getLogger('root.master.evaluator')
        logger.info('Best hyperparameter is followings:')
        yml = load_yaml(self.hp_result, self.dict_lock)
        logger.info(yml)

    def save(self) -> None:
        """Save current results to a file.

        Returns:
            None
        """
        yml = load_yaml(self.hp_result, self.dict_lock)
        path = self.ws / aiaccel.dict_result / aiaccel.file_final_result
        create_yaml(path, yml, self.dict_lock)
