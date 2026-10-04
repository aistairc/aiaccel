# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import pytest

from aiaccel.torch.utils.remove_absolute_path import is_absolute_path, remove_absolute_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/home/user/model.ckpt", True),
        (r"C:\Users\user\model.ckpt", True),
        (r"\\server\share\model.ckpt", True),
        ("checkpoints/model.ckpt", False),
    ],
)
def test_is_absolute_path(path: str, expected: bool) -> None:
    assert is_absolute_path(path) is expected


def test_remove_absolute_path() -> None:
    obj = {
        "relative": "checkpoints/model.ckpt",
        "absolute": "/home/user/model.ckpt",
        "nested": {"data_dir": "/data/dataset", "value": 1},
        "items": ["keep", "/remove/me"],
    }

    cleaned, removed = remove_absolute_path(obj)

    assert cleaned == {
        "relative": "checkpoints/model.ckpt",
        "nested": {"value": 1},
        "items": ["keep"],
    }
    assert removed == ["absolute", "nested.data_dir", "items[1]"]
