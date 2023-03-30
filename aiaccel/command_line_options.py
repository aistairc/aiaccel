from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CommandLineOptions:
    config: str
    resume: int | None
    clean: bool
    process_name: str = field(default="")
