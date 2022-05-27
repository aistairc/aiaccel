from aiaccel.util import aiaccel


def main(p):
    x1 = p["x1"]
    x2 = p["x2"]
    print (f"{x1}--type({type(x1)})")
    print (f"{x2}--type({type(x2)})")
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    print(y)
    return y


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
