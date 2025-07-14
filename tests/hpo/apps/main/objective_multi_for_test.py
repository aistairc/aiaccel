import argparse


def main(x1: float, x2: float) -> tuple[float, float]:
    obj1 = (x1 - 2) ** 2 + (x2 - 1) ** 2
    obj2 = x1 + x2

    return obj1, obj2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)

    args, unk_args = parser.parse_known_args()

    print(main(args.x1, args.x2))
