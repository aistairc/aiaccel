#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("out_filename", type=Path)
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)

    args = parser.parse_args()

    y = (args.x1**2) + (args.x2**2)

    with open(args.out_filename, "w") as f:
        f.write(f"{y:f}")
