from __future__ import annotations

import shutil

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

    def __init__(self):
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
        self.plot_config = {
            "height": 15,
            "colors": [self.line_colors[self.color_priority[0]], self.line_colors[self.color_priority[1]]],
        }
        self.plot_datas = [[]]

    def set_height(self, height: int) -> None:
        """Set the any height of vertical axis.

        Args:
            height (int): height of vertical axis.
        """
        self.plot_config["height"] = height

    # def set_width(self, width: int) -> None:
    #     """ Set the width of the horizontal axis.

    #     Args:
    #         wodth (int): width of the horizontal axis.
    #     """
    #     self.plot_config["width"] = width

    def set_colors(self, colors: list) -> None:
        """Set the color of line graph.

        Args:
            colors (list): The number of list item is same as the number of
                line.

        Raises:
            Exception: An out-of-target color is specified.
        """
        if not type(colors) == list:
            raise Exception

        self.plot_config["colors"] = []
        for c in colors:
            if c in self.line_colors:
                self.plot_config["colors"].append(self.line_colors[c])
            else:
                raise Exception

    def caption(self, *labels: tuple) -> None:
        """Set the any caption.

        Args:
            labels (tuple):
        """
        labels = list(labels)[0]
        for i in range(len(labels)):
            color = self.line_colors[self.color_priority[i]]
            print(f"{color}{list(labels)[i]}{reset}")

    def line_plot(self, *data: tuple) -> None:
        """Plot the any data.

        Args:
            data (tuple): Target data.

        Note:
            data = ([plot_data_1[], plot_data_2[],...)
        """
        terminal_size = shutil.get_terminal_size().columns
        plot_width_max = terminal_size - 15

        if self.plot_config["colors"] == []:
            colors = [self.color_priority[i] for i in range(len(data))]
            self.set_colors(colors)

        data = list(data)[0]
        self.plot_datas = []

        for i in range(len(data)):
            if type(data[i]) is not list:
                return
            if len(data[i]) == 0:
                return
            if None in data[i]:
                return
            if any(np.isnan(data[i])):
                message = "WARNING: result data has 'nan'"
                print(f"{yellow}{message}{reset}")
                return
            if float("inf") in data[i]:
                message = "WARNING: result data has 'inf'"
                print(f"{yellow}{message}{reset}")
                return
            if float("-inf") in data[i]:
                message = "WARNING: result data has '-inf'"
                print(f"{yellow}{message}{reset}")
                return

            if len(data[i]) >= plot_width_max:
                self.plot_datas.append(data[i][(len(data[i]) - plot_width_max) :])
            else:
                self.plot_datas.append(data[i])
        print(plot(series=self.plot_datas, cfg=self.plot_config))

    def sort(self, data: list, goal: str) -> list | None:
        """Sort the any data to maximize or minimize.

        Args:
            data (list): A swquential data.
            goal (str) : 'minimize' or 'maximize'.

        Returns:
            Sorted list, or None.
        """
        if not type(data) == list:
            return

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
