# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

#! /usr/bin/env python3

from argparse import ArgumentParser
import math
from pathlib import Path
import re

import torch


def get_score(filename: Path) -> float:
    """Extract the score from a filename.

    Args:
        filename (Path): A filename containing a score string.

    Returns:
        float: The extracted score. Returns math.inf if extraction fails.
    """
    score = re.search(r"score=([0-9.]+)", filename.stem)
    if score:
        return float(score.group(1))
    else:
        return math.inf


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("train_path", type=Path)
    parser.add_argument("--ckpt_name", type=str, default="merged.ckpt")
    parser.add_argument("--n_ckpt", type=int, default=10)
    parser.add_argument("--direction", type=str.lower, choices=["min", "max"], default="max")
    args = parser.parse_args()

    print("=" * 32)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 32)

    filename_list = list((args.train_path / "checkpoints").glob("*.ckpt"))
    filename_list = list(filter(lambda x: x.stem.startswith("epoch"), filename_list))
    filename_list.sort(key=get_score, reverse=args.direction == "max")

    print(args.train_path / "checkpoints")

    if len(filename_list) == 0:
        return

    state_dict = {}
    ckpt = {}
    for filename in filename_list[: args.n_ckpt][::-1]:
        print(filename)

        ckpt = torch.load(filename, map_location="cpu")

        for key, value in ckpt["state_dict"].items():
            if key not in state_dict:
                state_dict[key] = value / args.n_ckpt
            else:
                state_dict[key] += value / args.n_ckpt

    ckpt["state_dict"] = state_dict

    torch.save(ckpt, args.train_path / "checkpoints" / args.ckpt_name)


if __name__ == "__main__":
    main()
