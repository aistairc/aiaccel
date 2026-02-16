from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge import collect


def test_storage_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.storage")


def test_pairs_from_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    macro_db = tmp_path / "runs" / "train" / "000" / "macro" / "optuna.db"
    micro_db = tmp_path / "runs" / "train" / "000" / "micro" / "optuna.db"
    macro_db.parent.mkdir(parents=True, exist_ok=True)
    micro_db.parent.mkdir(parents=True, exist_ok=True)
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    def fake_load_best_params(path: Path, study_name: str | None) -> tuple[dict[str, float] | None, str | None]:
        if path == macro_db and study_name and "macro" in study_name:
            return {"x": 1.0}, None
        if path == micro_db and study_name and "micro" in study_name:
            return {"y": 2.0}, None
        return None, "missing_best"

    monkeypatch.setattr(collect, "_load_best_params", fake_load_best_params)
    results, diagnostics = collect._pairs_from_paths("demo", "train", [macro_db, micro_db])
    assert results == [(0, {"x": 1.0}, {"y": 2.0})]
    assert diagnostics == []


def test_pairs_from_layout_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scenario_output = tmp_path / "outputs" / "demo"
    macro_db = scenario_output / "runs" / "train" / "000" / "macro" / "optuna.db"
    micro_db = scenario_output / "runs" / "train" / "000" / "micro" / "optuna.db"
    macro_db.parent.mkdir(parents=True, exist_ok=True)
    micro_db.parent.mkdir(parents=True, exist_ok=True)
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    def fake_load_best_params(path: Path, study_name: str | None) -> tuple[dict[str, float] | None, str | None]:
        if path == macro_db and study_name and "macro" in study_name:
            return {"x": 1.0}, None
        if path == micro_db and study_name and "micro" in study_name:
            return {"y": 2.0}, None
        return None, "missing_best"

    monkeypatch.setattr(collect, "_load_best_params", fake_load_best_params)
    results, diagnostics = collect._pairs_from_paths(
        "demo",
        "train",
        list((scenario_output / "runs" / "train").rglob("optuna.db")),
    )
    assert results == [(0, {"x": 1.0}, {"y": 2.0})]
    assert diagnostics == []


def test_pairs_from_explicit_pairs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    macro_db = tmp_path / "macro.db"
    micro_db = tmp_path / "micro.db"
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    def fake_load_best_params(path: Path, _study_name: str | None) -> tuple[dict[str, float] | None, str | None]:
        if path == macro_db:
            return {"x": 1.0}, None
        if path == micro_db:
            return {"y": 2.0}, None
        return None, "missing_best"

    monkeypatch.setattr(collect, "_load_best_params", fake_load_best_params)
    results, diagnostics = collect._pairs_from_explicit_pairs([(macro_db, micro_db)])
    assert results == [(0, {"x": 1.0}, {"y": 2.0})]
    assert diagnostics == []


def test_pairs_from_paths_fallback_study_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    macro_db = tmp_path / "external" / "macro_result.db"
    micro_db = tmp_path / "external" / "micro_result.db"
    macro_db.parent.mkdir(parents=True, exist_ok=True)
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    monkeypatch.setattr(collect, "_pair_db_paths", lambda _paths: [])

    def fake_list_studies(path: Path) -> tuple[list[str], str | None]:
        if path == macro_db:
            return ["demo-train-macro-007"], None
        if path == micro_db:
            return ["demo-train-micro-007"], None
        return [], "missing_db"

    def fake_load_best_params(path: Path, study_name: str | None) -> tuple[dict[str, float] | None, str | None]:
        if path == macro_db and study_name == "demo-train-macro-007":
            return {"x": 1.5}, None
        if path == micro_db and study_name == "demo-train-micro-007":
            return {"y": 2.5}, None
        return None, "missing_best"

    monkeypatch.setattr(collect, "_list_studies", fake_list_studies)
    monkeypatch.setattr(collect, "_load_best_params", fake_load_best_params)

    results, diagnostics = collect._pairs_from_paths("demo", "train", [macro_db, micro_db])
    assert results == [(7, {"x": 1.5}, {"y": 2.5})]
    assert diagnostics == []
