import time
from aiaccel.util import opt


def func(p):
    x1 = p["x1"]
    x2 = p["x2"]
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    time.sleep(30)
    return y


if __name__ == "__main__":

    run = opt.Run()
    run.execute_and_report(func)
