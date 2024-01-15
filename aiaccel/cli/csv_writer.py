import csv
import os
from logging import StreamHandler, getLogger

from fasteners import InterProcessLock
from omegaconf.dictconfig import DictConfig

from aiaccel.storage import Storage
from aiaccel.util import TrialId
from aiaccel.workspace import Workspace

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())


class CsvWriter:
    """Provides report creation method.

    Args:
        config (DictConfig): Config object.

    Attributes:
        config (Config): Config object.
        workspace (Workspace): Workspace object.
        fp (Path): File path to the csv file named 'results.csv'.
        trialid (TrialId): TrialId object.
        storage (Storage): Storage object related to the workspace.
        lock_file (dict[str, str]): Dict object containing string path to lock.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.workspace = Workspace(self.config.generic.workspace)
        self.fp = self.workspace.retults_csv_file
        self.trialid = TrialId(self.config)
        self.storage = Storage(self.workspace.storage_file_path)
        self.lock_file = {"result_txt": str(self.workspace.lock / "result_txt")}

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

        Args:
            None

        Returns:
            None
        """
        data = []
        header = []

        trial_ids = self.storage.trial.get_all_trial_id()

        if trial_ids is None or len(trial_ids) == 0:
            return

        # write header
        example = self.storage.get_hp_dict(trial_ids[0])
        header.append("trial_id")
        for param in example["parameters"]:
            header.append(param["parameter_name"])
        header.append("objective")

        with InterProcessLock(self.lock_file["result_txt"]):
            with open(self.fp, "w") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(header)

        # write result data
        trial_id_str = [self._get_zero_padding_trial_id(trial_id) for trial_id in trial_ids]
        results = [self.storage.get_hp_dict(n) for n in trial_id_str]

        for contents in results:
            row = []
            row.append(str(contents["trial_id"]))
            for param in contents["parameters"]:
                row.append(param["value"])
            row.append(contents["result"])
            data.append(row)

        with InterProcessLock(self.lock_file["result_txt"]):
            with open(self.fp, "a") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerows(data)
