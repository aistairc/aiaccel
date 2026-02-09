from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.config import load_bridge_config
from aiaccel.hpo.modelbridge.execution import emit_commands
from aiaccel.hpo.modelbridge.prepare import prepare_train
from aiaccel.hpo.modelbridge.toolkit.io import read_json


def _config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> Any:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    return load_bridge_config(payload)


def test_emit_commands_deterministic(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _config(tmp_path, make_bridge_config)
    prepare_train(config)

    path1 = emit_commands(config, role="train", fmt="json")
    path2 = emit_commands(config, role="train", fmt="json")

    assert path1 == path2
    assert read_json(path1) == read_json(path2)

    shell_path = emit_commands(config, role="train", fmt="shell")
    content = shell_path.read_text(encoding="utf-8")
    assert "aiaccel-hpo optimize --config" in content


def test_emit_commands_requires_plan(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _config(tmp_path, make_bridge_config)
    with pytest.raises(FileNotFoundError):
        emit_commands(config, role="train", fmt="json")
