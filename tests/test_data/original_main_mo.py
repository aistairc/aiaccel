import numpy as np

from aiaccel.util import aiaccel


def main(p):
    x0 = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    x1 = np.array([p["x6"], p["x7"], p["x8"], p["x9"], p["x10"]])

    # Sphere
    y0 = np.sum(x0**2)
    y1 = np.sum(x1**2)
    return [y0, y1]


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
