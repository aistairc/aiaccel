# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets import CIFAR10


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory to store the CIFAR-10 dataset")
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    CIFAR10(root=root, train=True, download=True)
    CIFAR10(root=root, train=False, download=True)


if __name__ == "__main__":
    main()
