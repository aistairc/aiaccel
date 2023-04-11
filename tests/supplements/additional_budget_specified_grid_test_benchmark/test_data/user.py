import numpy as np

from aiaccel.util.aiaccel import Run

numbers = {"a": -5, "b": 0, "c": 5}


def main(p):
    x1 = np.log10(p["x1"]) * 10 + 5
    x2 = p["x2"]
    x3 = np.log10(p["x3"])
    x4 = p["x4"]
    x5 = numbers.get(p["x5"], 0)
    x6 = p["x6"]
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2


if __name__ == "__main__":
    run = Run()
    run.execute_and_report(main)
