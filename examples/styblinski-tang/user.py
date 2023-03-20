import numpy as np

from aiaccel.util import aiaccel


def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    t1 = np.sum(x**4)
    t2 = -16 * np.sum(x**2)
    t3 = 5 * np.sum(x)
    y = 0.5 * (t1 + t2 + t3)
    return float(y)


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
