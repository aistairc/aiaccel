import numpy as np

from aiaccel.util import aiaccel


def main(p):
    x = np.array([p["x1"], p["x2"], float(p["x3"])])
    y = np.sum(x ** 2)
    return y


if __name__ == "__main__":

    run = aiaccel.Run()
    run.execute_and_report(main)
