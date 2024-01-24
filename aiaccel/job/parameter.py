from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Parameter:
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.bounds = {param["name"]: param["bounds"] for param in self.parameters}
        self.initial = {param["name"]: param["initial"] for param in self.parameters}
        self.step = {param["name"]: param["step"] for param in self.parameters}
        self.log = {param["name"]: param["log"] for param in self.parameters}
        self.type = {param["name"]: param["type"] for param in self.parameters}
        self.values = {param["name"]: param["initial"] for param in self.parameters}
        self.names = [param["name"] for param in self.parameters]


if __name__ == "__main__":
    param = Parameter(
        [
            {
                "name": "x1",
                "type": "float",
                "bounds": [0, 10],
                "initial": 3,
                "step": 1,
                "log": False,
            },
            {
                "name": "x2",
                "type": "float",
                "bounds": [0, 10],
                "initial": 6,
                "step": 1,
                "log": False,
            },
        ]
    )

    assert param.bounds == {"x1": [0, 10], "x2": [0, 10]}
    assert param.initial == {"x1": 3, "x2": 6}
    assert param.step == {"x1": 1, "x2": 1}
    assert param.log == {"x1": False, "x2": False}
    assert param.type == {"x1": "float", "x2": "float"}
    assert param.names == ["x1", "x2"]

    assert param.values["x1"] == 3
    param.values["x1"] = 9
    assert param.values["x1"] == 9
    # print(param.value(["x1"]))
    print(param.bounds)
