"""Small IO helpers used across the pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path


def write_csv(path: Path, rows: Iterable[Mapping[str, float]]) -> Path:
    """Write ``rows`` to ``path`` as a CSV file and return the path."""

    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows_iter = list(rows)
    if not rows_iter:
        path.write_text("", encoding="utf-8")
        return path

    header = sorted(rows_iter[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows_iter:
            writer.writerow({key: row.get(key, "") for key in header})
    return path


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Serialise ``payload`` as JSON at ``path`` and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


__all__ = ["write_csv", "write_json"]
