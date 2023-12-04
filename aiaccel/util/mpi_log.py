from __future__ import annotations

from os import linesep
from pathlib import Path

from aiaccel.util.error import MpiError
from aiaccel.util.time import get_now_str


class MpiLog:
    def __init__(self, rank: int, processor: str, rank_log_path: Path) -> None:
        self._rank = rank
        self._processor = processor
        self._path = rank_log_path / f"{self._rank}.csv"
        self._tag: int | None = None

    def write(self, message: str, tag: int | None = None, stdout: bool = False) -> None:
        stime = get_now_str()
        if tag is not None:
            self._tag = tag
        if self._tag is None:
            raise MpiError("No tag value has been set.")
        sline = f'{self._tag},{self._rank},{self._processor},"{message}",{stime}{linesep}'
        with self._path.open(mode="a") as f:
            f.write(sline)
        if stdout:
            print(sline, end="")
