from typing import Any

from collections.abc import KeysView, Sequence
from pathlib import Path
from types import TracebackType

# This stub is just for passing mypy.
class File:
    def __init__(self, name: str | Path, mode: str = "r") -> None: ...
    def __enter__(self) -> File: ...
    def __exit__(
        self,
        ex_exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...
    def keys(self) -> KeysView[str]: ...
    def __getitem__(self, key: str) -> dict[str, Sequence[Any]]: ...
    def close(self) -> None: ...

#     def create_group(self, name: str) -> Group:
#         pass

# class Group:
#     def create_dataset(self, name: str, shape: Sequence[int] | None) -> Dataset:
#         pass

# class Dataset:
#     ...
