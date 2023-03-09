import csv
import os
from pathlib import Path
from logging import StreamHandler, getLogger

from fasteners import InterProcessLock

from aiaccel.config import Config
from aiaccel.storage import Storage
from aiaccel.util import TrialId

logger = getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.addHandler(StreamHandler())


class CsvWriter:
    """Provides report creation method.

    Args:
        config_path (str): Path to the config file.

    Attributes:
        config (Config): Config object.
        ws (Path): Directory path to the workspace.
        fp (Path): File path to the csv file named 'results.csv'.
        trialid (TrialId): TrialId object.
        storage (Storage): Storage object related to the workspace.
        lock_file (dict[str, str]): Dict object containing string path to lock.
    """

    def __init__(self, config_path: str) -> None:
        self.config = Config(config_path)
        self.ws = Path(self.config.workspace.get()).resolve()
        self.fp = self.ws / 'results.csv'
        self.trialid = TrialId(str(config_path))
        self.storage = Storage(self.ws)
        self.lock_file = {
            'result_txt': str(self.ws / 'lock' / 'result_txt')
        }

    def _get_zero_padding_trial_id(self, trial_id: int) -> str:
        """Gets string of trial id padded by zeros.

        Args:
            trial_id (int): Target trial id.

        Returns:
            str: Trial id padded by zeros.
        """
        return self.trialid.zero_padding_any_trial_id(trial_id)

    def create(self) -> None:
        """Creates repoprt.
        """
        data = []
        header = []

        trial_ids = self.storage.trial.get_all_trial_id()

        if trial_ids is None or len(trial_ids) == 0:
            return

        # write header
        example = self.storage.get_hp_dict(trial_ids[0])
        header.append('trial_id')
        for param in example['parameters']:
            header.append(param['parameter_name'])
        header.append('objective')

        with InterProcessLock(self.lock_file['result_txt']):
            with open(self.fp, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(header)

        # write result data
        trial_id_str = [self._get_zero_padding_trial_id(trial_id) for trial_id in trial_ids]
        results = [self.storage.get_hp_dict(n) for n in trial_id_str]

        for contents in results:
            row = []
            row.append(str(contents['trial_id']))
            for param in contents['parameters']:
                row.append(param['value'])
            row.append(contents['result'])
            data.append(row)

        with InterProcessLock(self.lock_file['result_txt']):
            with open(self.fp, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(data)
