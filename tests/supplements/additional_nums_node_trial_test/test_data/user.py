from __future__ import annotations

import numpy as np

from aiaccel.util.aiaccel import Run


def main(p: dict[str, float | int]) -> float:
    x = np.array(list(map(float, p.values())))
    y = sum(x ** 2)
    return y


if __name__ == '__main__':
    run = Run()
    run.execute_and_report(main)
