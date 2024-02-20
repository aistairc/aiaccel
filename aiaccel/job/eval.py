from __future__ import annotations

from typing import Any
import ast


def param_str_eval(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s
