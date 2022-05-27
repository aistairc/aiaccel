from aiaccel.util import aiaccel


def main(p):
    y = -1.0 * ((p["x1"]**2) - (4.0 * p["x1"]) + (p["x2"]**2) - p["x2"] - (p["x1"] * p["x2"]))
    return y


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
