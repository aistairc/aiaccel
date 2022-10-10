from pathlib import Path
from argparse import ArgumentParser

from numpy import maximum

from aiaccel.config import Config
from aiaccel.workspace import Workspace
from aiaccel.storage.storage import Storage


class Viewer:
    def __init__(self, config: Config):
        self.config_path = config.config_path
        self.workspace = Path(config.workspace.get()).resolve()
        self.storage = Storage(self.workspace)

    def view(self) -> None:
        """Print database information

        Returns:
            None
        """
        symbols = ['─', '╰', '╭', '╮', '╯', '│']
        infos = []
        len_margin = 2
        trial_ids = self.storage.trial.get_all_trial_id()

        for trial_id in trial_ids:

            start_time = self.storage.timestamp.get_any_trial_start_time(trial_id)
            end_time = self.storage.timestamp.get_any_trial_end_time(trial_id)
            status = self.storage.trial.get_any_trial_state(trial_id)
            job = self.storage.jobstate.get_any_trial_jobstate(trial_id)
            result = self.storage.result.get_any_trial_objective(trial_id)

            error = str(self.storage.error.get_any_trial_error(trial_id))
            if error == "None":
                error = ""
            else:
                if len(error) > 4:
                    error = error[0:3] + "..."

            # resume = "possible" if self.storage.serializer.is_exists_any_trial(trial_id) else ""

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

        max_column_width = []
        for info in infos:
            width = [len(info[key]) + len_margin for key in list(info.keys())]
            header_width = [len(key) + len_margin for key in list(info.keys())]
            if len(max_column_width) == 0:
                max_column_width = maximum(width, header_width).tolist()
            else:
                max_column_width = maximum(
                    max_column_width,
                    maximum(width, header_width).tolist()
                ).tolist()

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
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    args = parser.parse_args()

    config = Config(args.config)
    workspace = config.workspace.get()

    ws = Workspace(workspace)
    if ws.exists() is False:
        print(f"{workspace} is not found.")
        return

    viewer = Viewer(config)
    viewer.view()


if __name__ == "__main__":  # pragma: no cover
    main()
