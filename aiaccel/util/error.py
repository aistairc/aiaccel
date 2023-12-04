from __future__ import annotations

from inspect import getframeinfo, stack
from pathlib import Path
from traceback import format_exc

from aiaccel.util.time import get_now_str


class MpiError(Exception):
    def __init__(self, message: str, path: Path | None = None) -> None:
        strace = message
        try:
            super().__init__(message)
            stime = get_now_str()
            st = stack()[1]  # stack
            fr = st[0]  # frame
            fi = getframeinfo(fr)  # frame info
            strace += f"""
[{fi.lineno}, {fi.function}, {fi.filename}, {stime}]
{format_exc()}
"""
            if path is not None:
                with path.open("a") as f:
                    f.write(strace)
        finally:
            print(strace)
