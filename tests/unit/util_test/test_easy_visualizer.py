import shutil

import pytest

from aiaccel.util.easy_visualizer import EasyVisualizer


def test_easy_visualizer_init():
    assert EasyVisualizer().__init__() is None


def test_set_height():
    cplot = EasyVisualizer()

    assert cplot.plot_config["height"] == 15  # initial value
    cplot.set_height(5)
    assert cplot.plot_config["height"] == 5


def test_set_colors():
    cplot = EasyVisualizer()
    assert cplot.set_colors([]) is None
    assert cplot.set_colors(["red", "green"]) is None

    with pytest.raises(Exception):
        cplot.set_colors("red")

    with pytest.raises(Exception):
        cplot.set_colors(["darkred", "darkgreen"])


def test_caption():
    cplot = EasyVisualizer()
    assert cplot.caption(["aaa", "bbb"]) is None


def test_line_plot():
    cplot = EasyVisualizer()
    data = [1, 9, 2, 3, 4, 5]
    assert cplot.line_plot([data]) is None

    cplot.plot_config["colors"] = []
    assert cplot.line_plot([data]) is None

    no_data = []
    assert cplot.line_plot([no_data]) is None

    invalid_data = "123"
    assert cplot.line_plot([invalid_data]) is None

    none_data = [1, None, 2, 3, 4]
    assert cplot.line_plot([none_data]) is None

    nan_data = [1, float("nan"), 2, 3, 4]
    assert cplot.line_plot([nan_data]) is None

    inf_data = [1, float("inf"), 2, 3, 4]
    assert cplot.line_plot([inf_data]) is None

    inf_data = [1, float("-inf"), 2, 3, 4]
    assert cplot.line_plot([inf_data]) is None

    terminal_size = shutil.get_terminal_size().columns
    long_width_data = [i for i in range(terminal_size + 1)]
    assert cplot.line_plot([long_width_data]) is None


def test_sort():
    cplot = EasyVisualizer()

    invalid_data = 1
    goal = "maximize"
    assert cplot.sort(invalid_data, goal) is None

    data = [9, 1, 2, 3, 4, 5]
    goal = "maximize"
    assert cplot.sort(data, goal) == [9, 9, 9, 9, 9, 9]

    goal = "minimize"
    assert cplot.sort(data, goal) == [9, 1, 1, 1, 1, 1]

    goal = "invalid"
    assert cplot.sort(data, goal) == []
