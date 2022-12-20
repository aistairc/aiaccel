import copy
import logging
from pathlib import Path

from omegaconf.dictconfig import DictConfig

from aiaccel import dict_lock
from aiaccel import dict_verification
from aiaccel import extension_verification
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import create_yaml


class AbstractVerification(object):
    """An abstract class of verification.

    """

    def __init__(self, config: DictConfig) -> None:
        """Initial method for AbstractVerification.

        Args:
            config (ConfileWrapper): A configuration object.
        """
        # === Load config file===
        self.config = config
        self.ws = Path(self.config.generic.workspace).resolve()
        self.dict_lock = self.ws / dict_lock
        self.is_verified = None
        self.finished_loop = None
        self.condition = None
        self.verification_result = None
        self.load_verification_config()
        self.storage = Storage(self.ws)

    def verify(self) -> None:
        """Run a verification. The trigger to run a verification, is described
            in configuration file 'verification' > 'conditions'.

        Returns:
            None
        """
        if not self.is_verified:
            return

        # with fasteners.InterProcessLock(interprocess_lock_file(
        #         (self.ws / aiaccel.dict_hp), self.dict_lock)):
        #     hp_finished_files = get_file_hp_finished(self.ws)

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
        # with fasteners.InterProcessLock(
        #     interprocess_lock_file((self.ws / aiaccel.dict_hp), self.dict_lock)
        # ):
        #     hp_finished_files = get_file_hp_finished(self.ws)

        # best, best_file = get_best_parameter(
        #     hp_finished_files,
        #     self.config.goal.get(),
        #     self.dict_lock
        # )
        # self.verification_result[index]['best'] = best

        # if (
        #     best < self.condition[index]['minimum'] or
        #     best > self.condition[index]['maximum']
        # ):
        #     self.verification_result[index]['passed'] = False
        # else:
        #     self.verification_result[index]['passed'] = True

        # self.save(loop)

        best_trial = self.storage.get_best_trial_dict(self.config.optimize.goal.lower())

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
        self.is_verified = self.config.verification.is_verified
        self.condition = self.config.verification.condition
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

    def save(self, name: int) -> None:
        """Save current verifications result to a file.

        Args:
            name (int):

        Returns:
            None
        """
        if not self.is_verified:
            return None

        path = self.ws / dict_verification / f'{name}.{extension_verification}'
        create_yaml(path, self.verification_result, self.dict_lock)
        logger = logging.getLogger('root.master.verification')
        logger.info(f'Save verifiation file: {name}.{extension_verification}')
