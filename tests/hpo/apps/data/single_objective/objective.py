import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_filename", type=str)
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    args = parser.parse_args()

    y = (args.x1**2) - (4.0 * args.x1) + (args.x2**2) - args.x2 - (args.x1 * args.x2)

    with open(args.out_filename, "w") as f:
        f.write(f"{y}")


if __name__ == "__main__":
    main()
