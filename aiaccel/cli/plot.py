from argparse import ArgumentParser
from pathlib import Path

from aiaccel.config import Config
from aiaccel.storage import Storage
from aiaccel.util import EasyVisualizer


class Plotter:
    """Provides method to prints a graph on a terminal.

    Args:
        config (Config): Config object.

    Attributes:
        workspace (Path): Path to the workspace.
        storage (Storage): Storage object.
        goar (str): Goal of optimization ('minimize' or 'maximize').
        cplt (EasyVisualizer): EasyVisualizer object.
    """

    def __init__(self, config: Config) -> None:
        self.workspace = Path(config.workspace.get()).resolve()

        self.storage = Storage(self.workspace)
        self.goal = config.goal.get().lower()
        self.cplt = EasyVisualizer()

    def plot(self) -> None:
        """Retrieves information from the database and prints a graph on the
        terminal.

        Returns:
            None
        """
        objectives = self.storage.result.get_objectives()
        bests = self.storage.result.get_bests(self.goal)

        if len(objectives) == 0:
            print("Result data is empty")
            return

        if len(objectives) != len(bests):
            print("Invalid data")
            return

        self.cplt.set_colors(
            [
                "red",
                "green"
            ]
        )
        self.cplt.caption(
            [
                "objective",
                "best value"
            ]
        )
        self.cplt.line_plot(
            [
                objectives,
                bests
            ]
        )
        return


def main() -> None:  # pragma: no cover
    """Parses command line options and plots a graph on the terminal.
    """
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    args = parser.parse_args()

    config = Config(args.config)
    workspace = config.workspace.get()

    if Path(workspace).exists() is False:
        print(f"{workspace} is not found.")
        return

    plotter = Plotter(config)
    plotter.plot()


if __name__ == "__main__":  # pragma: no cover
    main()
