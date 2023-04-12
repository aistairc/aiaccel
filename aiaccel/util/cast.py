from __future__ import annotations

from typing import Any


def cast_y(
        y_value: Any, y_data_type: str | None) -> float | int | str:
    """Casts y to the appropriate data type.

    Args:
        y_value (Any): y value to be casted.
        y_data_type (str | None): Name of data type of objective value.

    Returns:
        float | int | str: Casted y value.

    Raises:
        TypeError: Occurs when given `y_data_type` is other than `float`,
                `int`, or `str`.
    """
    if y_data_type is None:
        y = y_value
    elif y_data_type.lower() == 'float':
        y = float(y_value)
    elif y_data_type.lower() == 'int':
        y = int(float(y_value))
    elif y_data_type.lower() == 'str':
        y = str(y_value)
    else:
        TypeError(f'{y_data_type} cannot be specified')

    return y
