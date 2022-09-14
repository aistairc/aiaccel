from aiaccel.argument import Arguments
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from pathlib import Path
from aiaccel.util.easy_visualizer import EasyVisualizer
from typing import Union


class Plotter:
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = config_path
        if type(self.config_path) is str:
            self.config_path = Path(self.config_path)
        self.config = Config(self.config_path)

        self.workspace = Path(self.config.workspace.get()).resolve()
        if self.workspace.exists() is False:
            print(f"{self.workspace} is not found.")
            raise ()

        self.storage = Storage(
            self.workspace,
            fsmode=self.config.filesystem_mode.get(),
            config_path=self.config_path
        )

        self.goal = self.config.goal.get().lower()
        self.cplt = EasyVisualizer()

    def plot(self) -> None:
        """Retrieves information from the database and prints a graph on the terminal.

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
    options = Arguments()
    if "config" not in options.keys():
        print("Specify the config file path with the --config option.")
        return

    plotter = Plotter(options['config'])
    plotter.plot()


if __name__ == "__main__":  # pragma: no cover
    main()
