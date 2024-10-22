import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("dst_filename", type=Path)
    parser.add_argument("--x", type=float)
    args = parser.parse_args()

    x = args.x

    y = x**2

    with open(args.dst_filename, "wb") as f:
        pkl.dump(y, f)

if __name__ == "__main__":
    main()
