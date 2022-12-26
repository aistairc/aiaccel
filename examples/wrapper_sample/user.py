import argparse


def main(x1, x2):
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x1', type=float)
    parser.add_argument('--x2', type=float)
    args = vars(parser.parse_known_args()[0])

    y = main(args["x1"], args["x2"])

    print(f"objective_y:{y}")
