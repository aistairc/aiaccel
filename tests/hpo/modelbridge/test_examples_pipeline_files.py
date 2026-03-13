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
    assert "MODELBRIDGE_APP := $(PROJECT_ROOT)/aiaccel/hpo/apps/modelbridge.py" in content


def test_makefile_calls_modelbridge_cli_directly() -> None:
    stage_map = {
        "prepare": '$(PYTHON) $(MODELBRIDGE_APP) prepare --config "$(CONFIG_PATH)" --workspace "$(WORKSPACE_PATH)"',
        "collect_train": '$(PYTHON) $(MODELBRIDGE_APP) collect --workspace "$(WORKSPACE_PATH)" --phase train',
        "collect_test": '$(PYTHON) $(MODELBRIDGE_APP) collect --workspace "$(WORKSPACE_PATH)" --phase test',
        "fit": '$(PYTHON) $(MODELBRIDGE_APP) fit-model --workspace "$(WORKSPACE_PATH)"',
        "evaluate": '$(PYTHON) $(MODELBRIDGE_APP) evaluate --workspace "$(WORKSPACE_PATH)"',
    }
    content = (_example_dir() / "Makefile").read_text(encoding="utf-8")
    for _, command in stage_map.items():
        assert command in content
    assert 'find "$$RUNS_DIR" -name "config.yaml" | LC_ALL=C sort | while IFS= read -r config_path; do \\' in content
    assert 'aiaccel-hpo optimize --config "$$config_path"; \\' in content
    assert "| $(STATE_DIR)" in content
    assert "SCRIPTS_DIR" not in content


def test_example_shell_wrappers_are_removed() -> None:
    scripts_dir = _example_dir() / "scripts"
    for name in ("prepare.sh", "run_hpo.sh", "collect.sh", "fit.sh", "evaluate.sh"):
        assert not (scripts_dir / name).exists()


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
