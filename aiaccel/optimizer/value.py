from __future__ import annotations

from typing import Any


class Value:
    def __init__(self, id: str, value: Any) -> None:  # noqa A002
        self.id: str = id
        self.value: Any = value
