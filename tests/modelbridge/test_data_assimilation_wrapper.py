# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "hpo" / "modelbridge" / "data_assimilation"


def _load_wrapper_module() -> ModuleType:
    wrapper_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "hpo"
        / "modelbridge"
        / "data_assimilation"
        / "mas_bench_wrapper.py"
    )
    wrapper_dir = str(wrapper_path.parent)
    sys.path.insert(0, wrapper_dir)
    spec = importlib.util.spec_from_file_location("mas_bench_wrapper", wrapper_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the class body executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(wrapper_dir)
    return module


def test_normalize_runtime_config_resolves_relative_paths_from_config_location(tmp_path: Path) -> None:
    module = _load_wrapper_module()
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "mas_bench_config.yaml"

    config = {
        "dataset_root": "../../datasets/masbench",
        "mas_bench_jar": "../bin/MAS-Bench.jar",
        "output_root": "./out",
    }

    normalized, output_root = module._normalize_runtime_config(
        config,
        config_path=config_path,
        output_root_override=None,
    )

    assert normalized["dataset_root"] == str((config_dir / "../../datasets/masbench").resolve())
    assert normalized["mas_bench_jar"] == str((config_dir / "../bin/MAS-Bench.jar").resolve())
    assert normalized["output_root"] == str((config_dir / "out").resolve())
    assert output_root == (config_dir / "out").resolve()


def test_normalize_runtime_config_prefers_cli_output_root_override(tmp_path: Path) -> None:
    module = _load_wrapper_module()
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "mas_bench_config.yaml"

    normalized, output_root = module._normalize_runtime_config(
        {"output_root": "./from_config"},
        config_path=config_path,
        output_root_override="../from_cli",
    )

    assert normalized["output_root"] == str((config_dir / "../from_cli").resolve())
    assert output_root == (config_dir / "../from_cli").resolve()


def test_resolve_config_path_can_fallback_to_cwd_for_existing_relative_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_wrapper_module()
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True)
    cwd = tmp_path / "repo"
    (cwd / "work" / "tmp").mkdir(parents=True)
    jar = cwd / "work" / "tmp" / "MAS-Bench.jar"
    jar.write_text("dummy", encoding="utf-8")
    monkeypatch.chdir(cwd)

    resolved = module._resolve_config_path("./work/tmp/MAS-Bench.jar", config_dir=config_dir, prefer_existing=True)

    assert resolved == str(jar.resolve())


def _scenario_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "micro_model": "FL1-1",
        "macro_model": "FS1-1",
        "train_scenarios": [{"id": "s0", "mock_target": 0.2}, {"id": "s1", "mock_target": 0.7}],
        "test_scenario": {"id": "t0", "mock_target": 0.4},
        "agent_sizes": {"naive": 1, "rational": 0, "ruby": 0},
        "seeds": {"micro": 10},
    }
    config.update(overrides)
    return config


def test_load_scenarios_applies_model_defaults_and_overrides() -> None:
    module = _load_wrapper_module()
    config = _scenario_config(
        train_scenarios=[{"id": "s0"}, {"id": "s1", "micro_model": "FL2-1", "macro_model": "FS2-1"}],
    )

    train, test = module._load_scenarios(config, mock=False)

    assert [(s.id, s.micro_model, s.macro_model) for s in train] == [
        ("s0", "FL1-1", "FS1-1"),
        ("s1", "FL2-1", "FS2-1"),
    ]
    assert test.id == "t0"


def test_load_scenarios_does_not_require_top_level_models_when_scenarios_declare_them() -> None:
    module = _load_wrapper_module()
    config = _scenario_config(
        train_scenarios=[
            {"id": "s0", "micro_model": "FL1-1", "macro_model": "FS1-1"},
            {"id": "s1", "micro_model": "FL2-1", "macro_model": "FS2-1"},
        ],
        test_scenario={"id": "t0", "micro_model": "FL3-1", "macro_model": "FS3-1"},
    )
    del config["micro_model"], config["macro_model"]

    train, test = module._load_scenarios(config, mock=False)

    assert [(s.micro_model, s.macro_model) for s in [*train, test]] == [
        ("FL1-1", "FS1-1"),
        ("FL2-1", "FS2-1"),
        ("FL3-1", "FS3-1"),
    ]


def test_load_scenarios_rejects_missing_model_without_top_level_default() -> None:
    module = _load_wrapper_module()
    config = _scenario_config(test_scenario={"id": "t0", "micro_model": "FL3-1"})
    del config["macro_model"]

    with pytest.raises(ValueError, match="'s0' has no 'macro_model' and no top-level 'macro_model' default"):
        module._load_scenarios(config, mock=True)


@pytest.mark.parametrize(
    ("overrides", "mock", "message"),
    [
        ({"scenarios": 2}, True, "replaced by 'train_scenarios'"),
        ({"train_scenarios": []}, True, "non-empty list"),
        ({"train_scenarios": [{"id": "s0"}, {"id": "s1"}]}, True, "mock_target"),
        ({}, False, "micro_model/macro_model"),
        ({"test_scenario": {"id": "s0"}}, True, "ids must be unique"),
    ],
)
def test_load_scenarios_rejects_identical_or_invalid_scenarios(
    overrides: dict[str, Any], mock: bool, message: str
) -> None:
    module = _load_wrapper_module()
    with pytest.raises(ValueError, match=message):
        module._load_scenarios(_scenario_config(**overrides), mock=mock)


def test_optimize_scenarios_passes_each_scenario_condition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_wrapper_module()
    calls: list[tuple[Any, ...]] = []

    class FakeStudy:
        def __init__(self, target: float) -> None:
            self.best_trial = type("Trial", (), {"params": {"sigma_naive0": target}})()

        def get_trials(self, **_: Any) -> list[object]:
            return [object()]

    def fake_run(*args: Any) -> FakeStudy:
        calls.append(args)
        return FakeStudy(args[6].mock_target)

    monkeypatch.setattr(module, "_run_aiaccel_optimization", fake_run)
    config = _scenario_config()
    train, _ = module._load_scenarios(config, mock=True)
    executor = module.MASBenchExecutor(config)

    results = module._optimize_scenarios(
        config,
        stage="micro",
        scenarios=train,
        executor=executor,
        mock=True,
        output_root=tmp_path,
        config_path=tmp_path / "config.yaml",
    )

    assert results == {"s0": {"sigma_naive0": 0.2}, "s1": {"sigma_naive0": 0.7}}
    # Study name, seed, and scenario condition differ per scenario.
    assert [call[0] for call in calls] == ["FL1-1-micro-s0-random-1-10", "FL1-1-micro-s1-random-1-11"]
    assert [call[3] for call in calls] == [10, 11]
    assert [call[6].mock_target for call in calls] == [0.2, 0.7]


def test_run_regression_pairs_results_by_scenario_id(tmp_path: Path) -> None:
    module = _load_wrapper_module()
    micro_best = {"s1": {"mu": 0.7}, "s0": {"mu": 0.2}, "s2": {"mu": 0.9}}
    macro_train_best = {"s0": {"mu": 0.2}, "s1": {"mu": 0.7}}

    payload = module._run_regression({}, micro_best, macro_train_best, {"mu": 0.4}, tmp_path)

    assert payload["train_scenarios"] == ["s0", "s1"]
    assert payload["predicted_micro"]["mu"] == pytest.approx(0.4)
    assert (tmp_path / "data_assimilation_regression.json").exists()


def test_bundled_config_runs_in_mock_mode_without_mas_bench_assets(tmp_path: Path) -> None:
    config = yaml.safe_load((EXAMPLE_DIR / "mas_bench_config.yaml").read_text(encoding="utf-8"))
    assert config["allow_mock"] is True

    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_DIR / "mas_bench_wrapper.py"),
            "--config",
            str(EXAMPLE_DIR / "mas_bench_config.yaml"),
            "--output-root",
            str(tmp_path / "out"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
        # The wrapper runs `aiaccel-job` by name; expose the console scripts of this interpreter.
        env={**os.environ, "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])},
    )

    summary = json.loads((tmp_path / "out" / "data_assimilation_summary.json").read_text(encoding="utf-8"))
    scenario_ids = [scenario["id"] for scenario in config["train_scenarios"]]
    assert summary["mock"] is True
    assert sorted(summary["micro_best"]) == sorted(scenario_ids)
    assert summary["regression"]["train_scenarios"] == scenario_ids
    assert len({json.dumps(params, sort_keys=True) for params in summary["micro_best"].values()}) == len(scenario_ids)
