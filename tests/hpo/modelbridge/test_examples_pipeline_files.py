from __future__ import annotations

from pathlib import Path

import yaml


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "hpo" / "modelbridge"


def _data_assimilation_dir() -> Path:
    return _example_dir() / "data_assimilation"


def test_makefile_is_orchestrator() -> None:
    makefile_path = _example_dir() / "Makefile"
    content = makefile_path.read_text(encoding="utf-8")
    assert "all: evaluate" in content
    assert "WORKSPACE_DIR ?= workspace" in content


def test_makefile_references_shell_wrappers_directly() -> None:
    stage_map = {
        "prepare": "bash $(SCRIPTS_DIR)/prepare.sh",
        "hpo_train": "bash $(SCRIPTS_DIR)/run_hpo.sh train",
        "hpo_test": "bash $(SCRIPTS_DIR)/run_hpo.sh test",
        "collect": "bash $(SCRIPTS_DIR)/collect.sh",
        "fit": "bash $(SCRIPTS_DIR)/fit.sh",
        "evaluate": "bash $(SCRIPTS_DIR)/evaluate.sh",
    }
    content = (_example_dir() / "Makefile").read_text(encoding="utf-8")
    for _, command in stage_map.items():
        assert command in content
    assert "| $(STATE_DIR)" in content


def test_scripts_call_python_tools() -> None:
    scripts_dir = _example_dir() / "scripts"
    assert "aiaccel/hpo/modelbridge/prepare.py" in (scripts_dir / "prepare.sh").read_text(encoding="utf-8")
    assert "aiaccel/hpo/modelbridge/collect.py" in (scripts_dir / "collect.sh").read_text(encoding="utf-8")
    assert "aiaccel/hpo/modelbridge/fit_model.py" in (scripts_dir / "fit.sh").read_text(encoding="utf-8")
    assert "aiaccel/hpo/modelbridge/evaluate.py" in (scripts_dir / "evaluate.sh").read_text(encoding="utf-8")


def test_config_contains_required_keys() -> None:
    config_path = _example_dir() / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["n_train"] >= 1
    assert config["n_test"] >= 1
    assert isinstance(config["objective_command"], list)
    assert "train_params" in config
    assert "test_params" in config


def test_data_assimilation_makefiles_use_wrapper_entrypoint() -> None:
    content = (_data_assimilation_dir() / "Makefile").read_text(encoding="utf-8")
    assert "mas_bench_wrapper.py" in content
    assert "--output-root" in content
    assert "modelbridge run" not in content

    assert not (_data_assimilation_dir() / "Makefile.template").exists()
    assert not (_data_assimilation_dir() / "aiaccel_job.sh").exists()
