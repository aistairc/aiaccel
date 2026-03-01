from __future__ import annotations

import importlib
from pathlib import Path

import pytest

cli_module = importlib.import_module("aiaccel.hpo.apps.modelbridge")


def test_step_commands_are_set() -> None:
    assert cli_module.STEP_COMMANDS == (
        "prepare",
        "collect",
        "fit-model",
        "evaluate",
    )


@pytest.mark.parametrize("command", cli_module.STEP_COMMANDS)
def test_each_step_command_has_handler(command: str) -> None:
    assert callable(cli_module._resolve_step_handler(command))


def test_removed_run_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli_module._parse_args(["run", "--config", "config.yaml"])


def test_cli_prepare_dispatches_to_prepare_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Path] = {}

    def fake_run_prepare(config_path: Path, workspace: Path) -> tuple[int, int]:
        observed["config_path"] = config_path
        observed["workspace"] = workspace
        return (0, 0)

    monkeypatch.setattr(cli_module.prepare, "run_prepare", fake_run_prepare)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("n_train: 0\nn_test: 0\n", encoding="utf-8")
    workspace = tmp_path / "workspace"

    with pytest.raises(SystemExit) as wrapped:
        cli_module.main(["prepare", "--config", str(config_path), "--workspace", str(workspace)])

    assert wrapped.value.code == 0
    assert observed["config_path"] == config_path
    assert observed["workspace"] == workspace


def test_cli_collect_dispatches_to_collect_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_run_collect(workspace: Path, phase: str) -> Path:
        observed["workspace"] = workspace
        observed["phase"] = phase
        return workspace / "pairs" / f"{phase}_pairs.csv"

    monkeypatch.setattr(cli_module.collect, "run_collect", fake_run_collect)
    workspace = tmp_path / "workspace"

    with pytest.raises(SystemExit) as wrapped:
        cli_module.main(["collect", "--workspace", str(workspace), "--phase", "train"])

    assert wrapped.value.code == 0
    assert observed["workspace"] == workspace
    assert observed["phase"] == "train"


def test_cli_exits_non_zero_when_handler_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_workspace: Path) -> Path | None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module.fit_model, "run_fit_model", fail)

    with pytest.raises(SystemExit) as wrapped:
        cli_module.main(["fit-model", "--workspace", "workspace"])

    assert wrapped.value.code == 1
