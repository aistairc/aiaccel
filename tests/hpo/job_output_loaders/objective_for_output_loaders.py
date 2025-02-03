from argparse import ArgumentParser
import json
from pathlib import Path
import pickle as pkl


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("dst_filename", type=Path)
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    parser.add_argument("--output_type", type=str)  # for test only
    args = parser.parse_args()

    x1, x2 = args.x1, args.x2

    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)

    if args.output_type == "json":
        with open(args.dst_filename, "w") as f:
            json.dump(y, f)
    elif args.output_type == "pkl":
        with open(args.dst_filename, "wb") as f:
            pkl.dump(y, f)
    elif args.output_type == "stdout":
        print(y)
    else:
        raise AssertionError(f"Invalid output type: {args.output_type}")


if __name__ == "__main__":
    main()
