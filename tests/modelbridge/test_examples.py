# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "hpo" / "modelbridge"


def test_modelbridge_readmes_start_with_pixi_getting_started() -> None:
    for readme_path in sorted(EXAMPLE_ROOT.rglob("README.md")):
        contents = readme_path.read_text(encoding="utf-8")
        assert "## Getting started" in contents
        assert "pixi run" in contents
        assert "pip install" not in contents


def test_makefiles_dispatch_work_through_configurable_job_command() -> None:
    for makefile_path in (EXAMPLE_ROOT / "basic" / "Makefile", EXAMPLE_ROOT / "data_assimilation" / "Makefile"):
        contents = makefile_path.read_text(encoding="utf-8")
        assert "cmd ?= aiaccel-job local" in contents
        assert "job_ops ?=" in contents
        assert "$(cmd) cpu $(job_ops)" in contents


def test_basic_makefile_uses_dedicated_modelbridge_console_script() -> None:
    contents = (EXAMPLE_ROOT / "basic" / "Makefile").read_text(encoding="utf-8")
    assert "AIACCEL_MODELBRIDGE ?= aiaccel-modelbridge" in contents
    assert "$(AIACCEL_HPO) modelbridge" not in contents
