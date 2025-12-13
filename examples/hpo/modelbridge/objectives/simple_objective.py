#! /usr/bin/env python3

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_filename", type=Path)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)

    args = parser.parse_args()

    # Simple sphere-like objective shifted
    micro_score = (args.x - 0.6) ** 2 + (args.y + 0.3) ** 2

    with open(args.out_filename, "w") as f:
        json.dump(micro_score, f)