from argparse import ArgumentParser
from typing import Any, Dict


def func(hparams: Dict[str, Any]) -> float:
    x1 = hparams["x1"]
    x2 = hparams["x2"]
    y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
    return float(y)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    args = parser.parse_args()

    hparams = {
        "x1": args.x1,
        "x2": args.x2,
    }

    print(func(hparams))
