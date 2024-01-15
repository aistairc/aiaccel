from __future__ import annotations

import shutil
from typing import Any

import numpy as np
from asciichartpy import (
    black,
    blue,
    cyan,
    darkgray,
    green,
    lightblue,
    lightcyan,
    lightgray,
    lightgreen,
    lightmagenta,
    lightred,
    lightyellow,
    magenta,
    plot,
    red,
    reset,
    white,
    yellow,
)


class EasyVisualizer:
    """Visualizer

    Example:
     ::

        cplt = EasyVisualizer()
        cplt.set_colors(["red"])
        cplt.line_plot([values: list])
    """

    def __init__(self) -> None:
        self.line_colors = {
            "black": black,
            "blue": blue,
            "magenta": magenta,
            "red": red,
            "white": white,
            "yellow": yellow,
            "cyan": cyan,
            "darkgray": darkgray,
            "green": green,
            "lightblue": lightblue,
            "lightcyan": lightcyan,
            "lightgray": lightgray,
            "lightgreen": lightgreen,
            "lightmagenta": lightmagenta,
            "lightred": lightred,
            "lightyellow": lightyellow,
        }
        self.color_priority = [
            "red",
            "green",
            "blue",
            "magenta",
            "white",
            "yellow",
            "cyan",
            "lightblue",
            "lightcyan",
            "lightgray",
            "lightgreen",
            "lightmagenta",
            "lightred",
            "lightyellow",
            "darkgray",
            "black",
        ]
        self.plot_config: dict[str, Any] = {
            "height": 15,
            "colors": [],
        }
        self.plot_data: list[list[float]]

    def set_height(self, height: int) -> None:
        """Set the any height of vertical axis.

        Args:
            height (int): height of vertical axis.

        Returns:
            None
        """
        self.plot_config["height"] = height

    # def set_width(self, width: int) -> None:
    #     """ Set the width of the horizontal axis.

    #     Args:
    #         wodth (int): width of the horizontal axis.
    #     """
    #     self.plot_config["width"] = width

    def set_colors(self, colors: list[Any]) -> None:
        """Set the color of line graph.

        Args:
            colors (list): The number of list item is same as the number of
                line.

        Raises:
            Exception: An out-of-target color is specified.
        """
        if not isinstance(colors, list):
            raise Exception

        self.plot_config["colors"] = []
        for c in colors:
            if c in self.line_colors:
                self.plot_config["colors"].append(self.line_colors[c])
            else:
                raise Exception

    def caption(self, labels: list[str]) -> None:
        """Set the any caption.

        Args:
            labels (tuple):
        """
        for label_id, label in enumerate(labels):
            color = self.line_colors[self.color_priority[label_id]]
            print(f"{color}{label}{reset}")

    def line_plot(self, data: list[list[float]]) -> None:
        """Plot the any data.

        Args:
            data (list[list[float]]): Target data.

        Note:
            data = ([plot_data_1[], plot_data_2[],...)
        """
        # Type validation
        if not isinstance(data, list):
            message = "ERROR: data must be list"
            print(f"{yellow}{message}{reset}")
            return

        terminal_size = shutil.get_terminal_size().columns
        plot_width_max = terminal_size - 15

        if self.plot_config["colors"] == []:
            colors = self.color_priority[: len(data)]
            self.set_colors(colors)

        self.plot_data = []
        for data_ in data:
            if not isinstance(data_, list):
                message = "WARNING: result data must be list"
                print(f"{yellow}{message}{reset}")
                return
            if len(data_) == 0:
                message = "WARNING: result data is empty"
                print(f"{yellow}{message}{reset}")
                return
            if None in data_:
                message = "WARNING: result data has 'None'"
                print(f"{yellow}{message}{reset}")
                return
            if np.nan in data_:
                message = "WARNING: result data has 'nan'"
                print(f"{yellow}{message}{reset}")
                return
            if float("inf") in data_:
                message = "WARNING: result data has 'inf'"
                print(f"{yellow}{message}{reset}")
                return
            if float("-inf") in data_:
                message = "WARNING: result data has '-inf'"
                print(f"{yellow}{message}{reset}")
                return
            self.plot_data.append(data_[max(len(data_) - plot_width_max, 0) :])

        print(plot(series=self.plot_data, cfg=self.plot_config))

    def sort(self, data: list[Any], goal: str) -> list[Any] | None:
        """Sort the any data to maximize or minimize.

        Args:
            data (list): A swquential data.
            goal (str) : 'minimize' or 'maximize'.

        Returns:
            Sorted list, or None.
        """
        if not isinstance(data, list):
            return None

        best_values = []

        if goal.lower() == "maximize":
            max_value = float("-inf")
            for d in data:
                if d > max_value:
                    max_value = d
                    best_values.append(d)
                else:
                    best_values.append(max_value)
            print(f"max_value:{max_value}")

        elif goal.lower() == "minimize":
            min_value = float("inf")
            for d in data:
                if d < min_value:
                    min_value = d
                    best_values.append(d)
                else:
                    best_values.append(min_value)
            print(f"min_value:{min_value}")

        else:
            # Usually Not reached.
            pass

        return best_values
