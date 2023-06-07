from __future__ import annotations

from typing import Union


def str_or_float_or_int(value: str | float | int) -> Union[str, float, int]:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def float_or_int(value: float | int) -> Union[float, int]:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError as e:
            raise e
