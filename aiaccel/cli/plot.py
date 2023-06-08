from argparse import ArgumentParser
from pathlib import Path

from omegaconf.dictconfig import DictConfig

from aiaccel.config import load_config
from aiaccel.storage.storage import Storage
from aiaccel.util.easy_visualizer import EasyVisualizer
from aiaccel.workspace import Workspace


class Plotter:
    """Provides method to prints a graph on a terminal.

    Args:
        config (DictConfig): A DictConfig object.

    Attributes:
        workspace (Workspace): Workspace object.
        storage (Storage): Storage object.
        goals (str): Goal of optimization ('minimize' or 'maximize').
        cplt (EasyVisualizer): EasyVisualizer object.
    """

    def __init__(self, config: DictConfig):
        self.workspace = Workspace(config.generic.workspace)
        self.storage = Storage(self.workspace.storage_file_path)
        self.goals = [item.value for item in config.optimize.goal]

        self.cplt = EasyVisualizer()

    def plot(self) -> None:
        """Retrieves information from the database and prints a graph on the
        terminal.
        """
        objectives = self.storage.result.get_objectives()
        if len(objectives) == 0:
            print("Result data is empty")
            return

        objectives = list(map(list, zip(*objectives)))

        if len(objectives) == 0:
            print("Result data is empty")
            return

        bests = self.storage.result.get_bests(self.goals)

        if len(objectives) != len(bests):
            print("Invalid data")
            return

        plot_data = []
        captions = []
        num_objectives = 0
        for goal_, objectives_ in zip(self.goals, objectives):
            current_best = float("inf") if goal_ == "minimize" else float("-inf")
            comparator = min if goal_ == "minimize" else max
            best_trajectory = []
            for objective in objectives_:
                current_best = comparator(current_best, objective)
                best_trajectory.append(current_best)
            plot_data.append(objectives_)
            plot_data.append(best_trajectory)
            captions.append(f"objective ({num_objectives})")
            captions.append(f"best value ({num_objectives})")
            num_objectives += 1

        self.cplt.caption(captions)
        self.cplt.line_plot(plot_data)


def main() -> None:  # pragma: no cover
    """Parses command line options and plots a graph on the terminal."""
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yml")
    args = parser.parse_args()

    config = load_config(args.config)

    if Path(config.generic.workspace).exists() is False:
        print(f"{config.generic.workspace} is not found.")
        return

    plotter = Plotter(config)
    plotter.plot()


if __name__ == "__main__":  # pragma: no cover
    main()
