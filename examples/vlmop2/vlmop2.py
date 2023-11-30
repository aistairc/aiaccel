import aiaccel
import numpy as np

def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"], p["x6"], p["x7"], p["x8"], p["x9"], p["x10"]])
    n = len(x)
    y1 = 1 - np.exp(-sum([(x[i] - 1 / np.sqrt(n))**2 for i in range(n)]))
    y2 = 1 - np.exp(-sum([(x[i] + 1 / np.sqrt(n))**2 for i in range(n)]))

    return [y1, y2]


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
