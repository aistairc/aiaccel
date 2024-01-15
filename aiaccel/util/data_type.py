from __future__ import annotations

from typing import Union


def str_or_float_or_int(value: str | float | int) -> Union[str, float, int]:
    """This indicates that the data is of type str, float, or int.

    Args:
        value (str | float | int): The value to be converted.

    Returns:
        Union[str, float, int]: The converted value.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def float_or_int(value: float | int) -> Union[float, int]:
    """This indicates that the data is of type float or int.

    Args:
        value (float | int): The value to be converted.

    Returns:
        Union[float, int]: The converted value.
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError as e:
            raise e
