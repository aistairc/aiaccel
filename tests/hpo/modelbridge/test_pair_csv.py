from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.common import write_csv
from aiaccel.hpo.modelbridge.pair_csv import pairs_to_rows, parse_pairs_csv


def test_pair_csv_roundtrip(tmp_path: Path) -> None:
    pairs = [
        (0, {"x": 1.0, "z": 3.0}, {"y": 2.0}),
        (1, {"x": 2.5, "z": 4.5}, {"y": 5.0}),
    ]
    csv_path = write_csv(tmp_path / "pairs.csv", pairs_to_rows(pairs))

    assert parse_pairs_csv(csv_path) == pairs


def test_pair_csv_parse_skips_invalid_run_id(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("run_id,macro_x,micro_y\nbad,1.0,2.0\n", encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="aiaccel.hpo.modelbridge.pair_csv")

    assert parse_pairs_csv(csv_path) == []
    assert any(str(csv_path) in record.getMessage() and "row=1" in record.getMessage() for record in caplog.records)


def test_pair_csv_parse_skips_partial_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "run_id,macro_x,micro_y\n"
        "0,1.0,\n"
        "1,,2.0\n"
        "2,3.0,4.0\n",
        encoding="utf-8",
    )

    assert parse_pairs_csv(csv_path) == [(2, {"x": 3.0}, {"y": 4.0})]
