import pathlib
import csv

from fasteners import InterProcessLock

from aiaccel.util.trial_id import TrialId
from aiaccel.config import Config
from aiaccel.storage.storage import Storage


class Reporter:
    def __init__(self, config_path: str):
        self.config = Config(config_path)

        self.ws = pathlib.Path(self.config.workspace.get()).resolve()
        self.fp = self.ws / 'results.csv'
        self.trialid = TrialId(config_path)
        self.storage = Storage(self.ws)
        self.lock_file = {
            'result_txt': str(self.ws / 'lock' / 'result_txt')
        }

    def get_zero_padding_trial_id(self, trial_id: int):
        return self.trialid.zero_padding_any_trial_id(trial_id)

    def create(self):
        data = []
        header = []

        finished = self.storage.trial.get_finished()
        if len(finished) == 0:
            print('No buffer.')
            return

        # write header
        example = self.storage.get_hp_dict(self.get_zero_padding_trial_id(finished[0]))
        header.append('trial_id')
        for param in example['parameters']:
            header.append(param['parameter_name'])
        header.append('objective')

        with InterProcessLock(self.lock_file['result_txt']):
            with open(self.fp, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(header)

        # write result data
        trial_id_str = [self.get_zero_padding_trial_id(trial_id) for trial_id in finished]
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
