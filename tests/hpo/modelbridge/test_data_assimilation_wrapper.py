from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _load_wrapper_module() -> ModuleType:
    wrapper_path = (
        Path(__file__).resolve().parents[3]
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
