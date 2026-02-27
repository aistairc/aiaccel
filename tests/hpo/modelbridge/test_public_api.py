from __future__ import annotations

import aiaccel.hpo.modelbridge as modelbridge
from aiaccel.hpo.modelbridge import collect, evaluate, fit_model, prepare


def test_package_root_exports_spec17_modules() -> None:
    assert modelbridge.__all__ == [
        "collect",
        "evaluate",
        "fit_model",
        "prepare",
    ]
    assert modelbridge.collect is collect
    assert modelbridge.evaluate is evaluate
    assert modelbridge.fit_model is fit_model
    assert modelbridge.prepare is prepare
