#! /usr/bin/env python3

import argparse
import json
import numpy as np
from pathlib import Path

def sphere(vec):
    return float(np.sum(vec**2))

def rastrigin(vec):
    return float(10 * vec.size + np.sum(vec**2 - 10.0 * np.cos(2 * np.pi * vec)))

def griewank(vec):
    denom = np.sqrt(np.arange(1, vec.size + 1, dtype=float))
    return float(np.sum(vec**2) / 4000.0 - np.prod(np.cos(vec / denom)) + 1.0)

FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "griewank": griewank,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_filename", type=Path)
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--x1", type=float, default=0.0)
    parser.add_argument("--x2", type=float, default=0.0)
    
    args = parser.parse_args()
    
    if args.function not in FUNCTIONS:
        raise ValueError(f"Unknown function: {args.function}")
        
    vec = np.array([args.x1, args.x2])
    score = FUNCTIONS[args.function](vec)
    
    with open(args.out_filename, "w") as f:
        json.dump(score, f)
