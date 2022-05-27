from aiaccel.util import aiaccel
import numpy as np


def main(p):
    
    print("\n\n{}\n\n".format("CATEGOLICAL_TEST"))

    cd = {
        "red": 1,
        "green": 2,
        "blue": 3,
        "yellow": 4
    }

    x = []
    x = np.array([cd[p["x1"]], cd[p["x2"]]])
    y = np.sum(x**2)

    return float(y)


if __name__ == "__main__":

    run = aiaccel.Run()
    run.execute_and_report(main)
