import argparse


def main(x1: float, x2: float) -> float:
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)

    args, unk_args = parser.parse_known_args()

    print(main(args.x1, args.x2))
