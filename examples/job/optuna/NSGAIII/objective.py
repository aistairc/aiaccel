import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("dst_filename", type=Path)
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    parser.add_argument("--x3", type=float)
    args = parser.parse_args()

    x1 = args.x1
    x2 = args.x2
    x3 = args.x3

    f1 = x1**2 + x2**2 + x3**2  # 最小化
    f2 = (x1 - 1) ** 2 + (x2 - 1) ** 2 + (x3 - 1) ** 2  # 最小化
    f3 = x1 + x2 + x3  # 最大化

    with open(args.dst_filename, "wb") as f:
        pkl.dump((f1, f2, f3), f)


if __name__ == "__main__":
    main()
