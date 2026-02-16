from __future__ import annotations

import aiaccel.hpo.modelbridge as modelbridge
from aiaccel.hpo.modelbridge.api import load_config, run
from aiaccel.hpo.modelbridge.common import PipelineResult
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.pipeline import run_pipeline


def test_package_root_exports_minimal_stable_api() -> None:
    assert modelbridge.__all__ == [
        "BridgeConfig",
        "PipelineResult",
        "load_bridge_config",
        "load_config",
        "run",
        "run_pipeline",
    ]
    assert modelbridge.BridgeConfig is BridgeConfig
    assert modelbridge.PipelineResult is PipelineResult
    assert modelbridge.load_bridge_config is load_bridge_config
    assert modelbridge.load_config is load_config
    assert modelbridge.run is run
    assert modelbridge.run_pipeline is run_pipeline
