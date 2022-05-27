from aiaccel.util import aiaccel


def main(p):
    return 1 / 0


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
