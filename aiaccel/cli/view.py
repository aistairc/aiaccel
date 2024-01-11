from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from numpy import maximum
from omegaconf.dictconfig import DictConfig

from aiaccel.config import load_config
from aiaccel.storage.storage import Storage
from aiaccel.workspace import Workspace


class Viewer:
    """Provides a method to print database information.

    Args:
        config (Config): Config object.

    Attributes:
        workspace (Workspace): Workspace object.
        storage (Storage): Storage object.
    """

    def __init__(self, config: DictConfig) -> None:
        self.workspace = Workspace(config.generic.workspace)
        self.storage = Storage(self.workspace.storage_file_path)

    def view(self) -> None:
        """Print database information

        Returns:
            None
        """
        symbols = ["─", "╰", "╭", "╮", "╯", "│"]
        infos = []
        len_margin = 2
        trial_ids = self.storage.trial.get_all_trial_id()

        for trial_id in trial_ids:
            start_time = self.storage.timestamp.get_any_trial_start_time(trial_id)
            end_time = self.storage.timestamp.get_any_trial_end_time(trial_id)
            status = self.storage.trial.get_any_trial_state(trial_id)
            job = self.storage.jobstate.get_any_trial_jobstate(trial_id)
            result = self.storage.result.get_any_trial_objective(trial_id)

            error_file = self.workspace.get_error_output_file(trial_id)
            error = ""
            if error_file.exists():
                with open(error_file, "r") as f:
                    error = f.read()
                if len(error) > 4:
                    error = error[0:3] + "..."

            infos.append(
                {
                    "trial_id": str(trial_id),
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "status": str(status),
                    "job": str(job),
                    "result": str(result),
                    "error": error,
                    # "resume": resume
                }
            )

        max_column_width: list[Any] = []
        for info in infos:
            width = [len(info[key]) + len_margin for key in list(info.keys())]
            header_width = [len(key) + len_margin for key in list(info.keys())]
            if len(max_column_width) == 0:
                max_column_width = maximum(width, header_width).tolist()
            else:
                max_column_width = maximum(max_column_width, maximum(width, header_width).tolist()).tolist()

        max_width = sum(max_column_width)
        print("\n")
        print(f"{symbols[2]}{symbols[0] * max_width}{symbols[3]}")

        # Header
        header = list(infos[0].keys())
        print(f"{symbols[5]}", end="")
        for i in range(len(header)):
            margin = " " * (max_column_width[i] - len(header[i]))
            print(f"{header[i]}{margin}", end="")
        print(f"{symbols[5]}")

        # Separator
        print(f"{symbols[5]}{symbols[0] * sum(max_column_width)}{symbols[5]}")

        for info in infos:
            print(f"{symbols[5]}", end="")
            keys = list(info.keys())
            for i in range(len(keys)):
                margin = " " * (max_column_width[i] - len(info[keys[i]]))
                print(f"{info[keys[i]]}{margin}", end="")
            print(f"{symbols[5]}")
        print(f"{symbols[1]}{symbols[0] * max_width}{symbols[4]}")


def main() -> None:  # pragma: no cover
    """Parses command line options and prints database information on
    theterminal.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yml")
    args = parser.parse_args()

    config: DictConfig = load_config(args.config)
    workspace = config.generic.workspace

    ws = Workspace(workspace)
    if ws.exists() is False:
        print(f"{workspace} is not found.")
        return

    viewer = Viewer(config)
    viewer.view()


if __name__ == "__main__":  # pragma: no cover
    main()
