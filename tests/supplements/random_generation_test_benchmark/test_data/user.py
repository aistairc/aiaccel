from aiaccel.util import aiaccel
import numpy as np


def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"], p["x6"], p["x7"], p["x8"], p["x9"]])
    y = np.sum(x ** 2)

    return y


if __name__ == "__main__":

    run = aiaccel.Run()
    run.execute_and_report(main)
