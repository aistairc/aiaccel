import pickle as pkl

from argparse import ArgumentParser
from pathlib import Path
import numpy as np

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("dst_filename", type=Path)
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    args = parser.parse_args()

    x1, x2 = args.x1, args.x2

    y = x1**2 + x2

    with open(args.dst_filename, "wb") as f:
        pkl.dump(y, f)


if __name__ == "__main__":
    main()
