import csv
import os
import pathlib
from logging import StreamHandler, getLogger

from fasteners import InterProcessLock

from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.trialid import TrialId

logger = getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.addHandler(StreamHandler())


class CreationReport:
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self.ws = pathlib.Path(self.config.workspace.get()).resolve()
        self.fp = self.ws / 'results.csv'
        self.trialid = TrialId(str(config_path))
        self.storage = Storage(self.ws)
        self.lock_file = {
            'result_txt': str(self.ws / 'lock' / 'result_txt')
        }

    def get_zero_padding_trial_id(self, trial_id: int):
        return self.trialid.zero_padding_any_trial_id(trial_id)

    def create(self):
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
        trial_id_str = [self.get_zero_padding_trial_id(trial_id) for trial_id in trial_ids]
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
