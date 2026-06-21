# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import yaml


def _modelbridge_examples_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "hpo" / "modelbridge"


def _example_dir() -> Path:
    return _modelbridge_examples_dir() / "basic"


def _data_assimilation_dir() -> Path:
    return _modelbridge_examples_dir() / "data_assimilation"


def test_modelbridge_examples_are_split_by_workflow() -> None:
    example_root = _modelbridge_examples_dir()
    assert (example_root / "basic").is_dir()
    assert (example_root / "data_assimilation").is_dir()
    assert (example_root / "README.md").exists()


def test_makefile_is_orchestrator() -> None:
    makefile_path = _example_dir() / "Makefile"
    content = makefile_path.read_text(encoding="utf-8")
    assert "all: evaluate" in content
    assert "WORKSPACE_DIR ?= workspace" in content
    assert "AIACCEL_HPO ?= aiaccel-hpo" in content
    assert "AIACCEL_WORKFLOW ?= aiaccel-workflow" in content
    assert "MODELBRIDGE_CMD ?= $(AIACCEL_HPO) modelbridge" in content
    assert "WORKFLOW_STAGES_MK := $(shell $(AIACCEL_WORKFLOW) template stages.mk)" in content
    assert "include $(WORKFLOW_STAGES_MK)" in content
    assert "min_stage := 1" in content
    assert "max_stage := 6" in content
    assert "stage1_dependencies := $(STATE_DIR)/01_prepare.done" in content
    assert "stage6_dependencies := $(STATE_DIR)/06_evaluate.done" in content
    assert "prepare: stage1" in content
    assert "evaluate: stage6" in content


def test_makefile_calls_modelbridge_cli_directly() -> None:
    stage_map = {
        "prepare": '$(MODELBRIDGE_CMD) prepare --config "$(CONFIG_PATH)" --workspace "$(WORKSPACE_PATH)"',
        "collect_train": '$(MODELBRIDGE_CMD) collect --workspace "$(WORKSPACE_PATH)" --phase train',
        "collect_test": '$(MODELBRIDGE_CMD) collect --workspace "$(WORKSPACE_PATH)" --phase test',
        "fit": '$(MODELBRIDGE_CMD) fit-model --workspace "$(WORKSPACE_PATH)"',
        "evaluate": '$(MODELBRIDGE_CMD) evaluate --workspace "$(WORKSPACE_PATH)"',
    }
    content = (_example_dir() / "Makefile").read_text(encoding="utf-8")
    for _, command in stage_map.items():
        assert command in content
    assert 'find "$$RUNS_DIR" -name "config.yaml" | LC_ALL=C sort | while IFS= read -r config_path; do \\' in content
    assert '$(AIACCEL_HPO) optimize --config "$$config_path"; \\' in content
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


def test_basic_abci_docs_and_config_use_pip_entrypoints() -> None:
    readme_content = (_example_dir() / "README.md").read_text(encoding="utf-8")
    job_config_content = (_example_dir() / "config" / "job_config_abci.yaml").read_text(encoding="utf-8")

    assert 'python -m pip install -e ".[dev,github-actions,modelbridge]"' in readme_content
    assert "aiaccel-hpo modelbridge" in readme_content
    assert "aiaccel-workflow" in readme_content
    assert "aiaccel-config" in readme_content
    assert "modelbridge_venv" in job_config_content
    assert "MODELBRIDGE_VENV" in job_config_content
    assert "pixi" not in readme_content.lower()
    assert "pixi" not in job_config_content.lower()
    assert "path_to_env" not in job_config_content
    assert "path_to_venv" not in job_config_content


def test_data_assimilation_makefiles_use_wrapper_entrypoint() -> None:
    content = (_data_assimilation_dir() / "Makefile").read_text(encoding="utf-8")
    assert "mas_bench_wrapper.py" in content
    assert "--output-root" in content
    assert "modelbridge run" not in content

    assert not (_data_assimilation_dir() / "Makefile.template").exists()
    assert not (_data_assimilation_dir() / "aiaccel_job.sh").exists()
