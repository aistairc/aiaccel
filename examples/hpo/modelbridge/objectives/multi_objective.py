#! /usr/bin/env python3

from numpy.typing import NDArray

import argparse
import json
from pathlib import Path

import numpy as np


def sphere(vec: NDArray[np.float64]) -> float:
    return float(np.sum(vec**2))


def rastrigin(vec: NDArray[np.float64]) -> float:
    return float(10 * vec.size + np.sum(vec**2 - 10.0 * np.cos(2 * np.pi * vec)))


def griewank(vec: NDArray[np.float64]) -> float:
    denom = np.sqrt(np.arange(1, vec.size + 1, dtype=float))
    return float(np.sum(vec**2) / 4000.0 - np.prod(np.cos(vec / denom)) + 1.0)


FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "griewank": griewank,
}

FUNCTION_IDS = {
    0: "sphere",
    1: "rastrigin",
    2: "griewank",
}


def resolve_function_name(function: str | None, function_id: float | None) -> str:
    if function is not None:
        return function
    if function_id is None:
        raise ValueError("Either --function or --function_id must be specified.")
    resolved_id = int(round(function_id))
    if resolved_id not in FUNCTION_IDS:
        raise ValueError(f"Unknown function_id: {function_id}")
    return FUNCTION_IDS[resolved_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_filename", type=Path)
    parser.add_argument("--function", type=str, default=None)
    parser.add_argument("--function_id", type=float, default=None)
    parser.add_argument("--x1", type=float, default=0.0)
    parser.add_argument("--x2", type=float, default=0.0)

    args = parser.parse_args()

    function_name = resolve_function_name(args.function, args.function_id)
    if function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}")

    vec = np.array([args.x1, args.x2])
    score = FUNCTIONS[function_name](vec)

    with open(args.out_filename, "w") as f:
        json.dump(score, f)
