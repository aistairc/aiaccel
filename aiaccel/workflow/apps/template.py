# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("mkfilename", type=str, help="File name of makefile")
    args, unk_args = parser.parse_known_args()

    filepath = Path(__file__).parent / "templete_files" / args.mkfilename
    print(filepath)


if __name__ == "__main__":
    main()
