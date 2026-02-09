from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import optuna
import pytest

from aiaccel.hpo.modelbridge import storage
from aiaccel.hpo.modelbridge.config import ScenarioConfig


def test_load_best_params(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class DummyTrial:
        params = {"x": 1.23}

    class DummyStudy:
        best_trial = DummyTrial()

    def fake_load_study(*_args: object, **_kwargs: object) -> DummyStudy:
        return DummyStudy()

    monkeypatch.setattr(optuna, "load_study", fake_load_study)
    result = storage.load_best_params(tmp_path / "optuna.db", "study")
    assert result == {"x": 1.23}


def test_scan_db_paths_for_pairs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    macro_db = tmp_path / "runs" / "train" / "000" / "macro" / "optuna.db"
    micro_db = tmp_path / "runs" / "train" / "000" / "micro" / "optuna.db"
    macro_db.parent.mkdir(parents=True, exist_ok=True)
    micro_db.parent.mkdir(parents=True, exist_ok=True)
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    scenario = MagicMock(spec=ScenarioConfig)
    scenario.name = "demo"

    def fake_load_best_params(_path: Path, study_name: str | None) -> dict[str, float]:
        if study_name and "macro" in study_name:
            return {"x": 1.0}
        return {"y": 2.0}

    monkeypatch.setattr(storage._DEFAULT_STORE, "load_best_params", fake_load_best_params)

    results = storage.scan_db_paths_for_pairs(scenario, "train", [macro_db, micro_db])
    assert results == [(0, {"x": 1.0}, {"y": 2.0})]


def test_scan_runs_for_pairs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scenario_dir = tmp_path / "outputs" / "demo"
    macro_db = scenario_dir / "runs" / "train" / "000" / "macro" / "optuna.db"
    micro_db = scenario_dir / "runs" / "train" / "000" / "micro" / "optuna.db"
    macro_db.parent.mkdir(parents=True, exist_ok=True)
    micro_db.parent.mkdir(parents=True, exist_ok=True)
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    scenario = MagicMock(spec=ScenarioConfig)
    scenario.name = "demo"

    def fake_load_best_params(_path: Path, study_name: str | None) -> dict[str, float]:
        if study_name and "macro" in study_name:
            return {"x": 1.0}
        return {"y": 2.0}

    monkeypatch.setattr(storage._DEFAULT_STORE, "load_best_params", fake_load_best_params)
    results = storage.scan_runs_for_pairs(scenario, scenario_dir, "train")
    assert results == [(0, {"x": 1.0}, {"y": 2.0})]


def test_load_pairs_from_db_pairs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    macro_db = tmp_path / "macro.db"
    micro_db = tmp_path / "micro.db"
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    scenario = MagicMock(spec=ScenarioConfig)
    scenario.name = "demo"

    def fake_load_best_params(path: Path, _study_name: str | None) -> dict[str, float]:
        if path == macro_db:
            return {"x": 1.0}
        return {"y": 2.0}

    monkeypatch.setattr(storage._DEFAULT_STORE, "load_best_params", fake_load_best_params)
    results = storage.load_pairs_from_db_pairs(scenario, "train", [(macro_db, micro_db)])
    assert results == [(0, {"x": 1.0}, {"y": 2.0})]


def test_scan_db_paths_for_pairs_fallback_study_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    macro_db = tmp_path / "external" / "macro_result.db"
    micro_db = tmp_path / "external" / "micro_result.db"
    macro_db.parent.mkdir(parents=True, exist_ok=True)
    macro_db.write_text("", encoding="utf-8")
    micro_db.write_text("", encoding="utf-8")

    scenario = MagicMock(spec=ScenarioConfig)
    scenario.name = "demo"

    monkeypatch.setattr(storage, "_pair_db_paths", lambda _paths: [])

    def fake_list_studies(path: Path) -> list[str]:
        if path == macro_db:
            return ["demo-train-macro-007"]
        if path == micro_db:
            return ["demo-train-micro-007"]
        return []

    def fake_load_best_params(path: Path, study_name: str | None) -> dict[str, float] | None:
        if path == macro_db and study_name == "demo-train-macro-007":
            return {"x": 1.5}
        if path == micro_db and study_name == "demo-train-micro-007":
            return {"y": 2.5}
        return None

    monkeypatch.setattr(storage._DEFAULT_STORE, "list_studies", fake_list_studies)
    monkeypatch.setattr(storage._DEFAULT_STORE, "load_best_params", fake_load_best_params)

    results = storage.scan_db_paths_for_pairs(scenario, "train", [macro_db, micro_db])
    assert results == [(7, {"x": 1.5}, {"y": 2.5})]
