from __future__ import annotations

import aiaccel.hpo.modelbridge as modelbridge
from aiaccel.hpo.modelbridge.api import (
    collect_eval_step,
    collect_train_step,
    evaluate_model_step,
    fit_regression_step,
    hpo_eval_step,
    hpo_train_step,
    load_config,
    prepare_eval_step,
    prepare_train_step,
    publish_summary_step,
)
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config


def test_package_root_exports_minimal_stable_api() -> None:
    assert modelbridge.__all__ == [
        "BridgeConfig",
        "collect_eval_step",
        "collect_train_step",
        "evaluate_model_step",
        "fit_regression_step",
        "hpo_eval_step",
        "hpo_train_step",
        "load_config",
        "load_bridge_config",
        "prepare_eval_step",
        "prepare_train_step",
        "publish_summary_step",
    ]
    assert modelbridge.BridgeConfig is BridgeConfig
    assert modelbridge.collect_eval_step is collect_eval_step
    assert modelbridge.collect_train_step is collect_train_step
    assert modelbridge.evaluate_model_step is evaluate_model_step
    assert modelbridge.fit_regression_step is fit_regression_step
    assert modelbridge.hpo_eval_step is hpo_eval_step
    assert modelbridge.hpo_train_step is hpo_train_step
    assert modelbridge.load_bridge_config is load_bridge_config
    assert modelbridge.load_config is load_config
    assert modelbridge.prepare_eval_step is prepare_eval_step
    assert modelbridge.prepare_train_step is prepare_train_step
    assert modelbridge.publish_summary_step is publish_summary_step
