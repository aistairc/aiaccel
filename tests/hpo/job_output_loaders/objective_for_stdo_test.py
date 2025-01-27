from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    args = parser.parse_args()

    x1, x2 = args.x1, args.x2

    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)

    print(y)


if __name__ == "__main__":
    main()
