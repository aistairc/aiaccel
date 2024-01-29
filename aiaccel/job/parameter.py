from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optuna.trial import Trial


@dataclass
class Parameter:
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.names = [param["name"] for param in self.parameters]
        self.bounds = {param["name"]: param["bounds"] for param in self.parameters}
        self.initial = {
            param["name"]: param.get("initial", None) for param in self.parameters
        }
        self.step = {
            param["name"]: param.get("step", None) for param in self.parameters
        }
        self.log = {param["name"]: param.get("log", False) for param in self.parameters}
        self.type = {
            param["name"]: param.get("type", "float") for param in self.parameters
        }
        self.values = {param["name"]: None for param in self.parameters}

    def update_values(self, param_name, new_value):
        if param_name in self.values:
            if self.type[param_name] == "int":
                self.values[param_name] = int(new_value)
            elif self.type[param_name] == "float":
                self.values[param_name] = float(new_value)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Parameter with name '{param_name}' not found.")


def suggest_hyperparameter(trial: Trial, parameter: Parameter) -> Parameter:
    """Suggest hyperparameters."""

    for name in parameter.names:
        if parameter.type[name] == "int":
            parameter.values[name] = trial.suggest_int(
                name,
                parameter.bounds[name][0],
                parameter.bounds[name][1],
                log=parameter.log[name],
            )
        elif parameter.type[name] == "float":
            parameter.values[name] = trial.suggest_float(
                name,
                parameter.bounds[name][0],
                parameter.bounds[name][1],
                log=parameter.log[name],
            )
        elif parameter.type[name] == "categorical":
            parameter.values[name] = trial.suggest_categorical(
                name, parameter.bounds[name]
            )
        else:
            raise NotImplementedError
    return parameter


# if __name__ == "__main__":
#     param = Parameter(
#         [
#             {
#                 "name": "x1",
#                 "type": "float",
#                 "bounds": [0, 10],
#                 "initial": 3,
#                 "step": 1,
#                 "log": False,
#             },
#             {
#                 "name": "x2",
#                 "type": "float",
#                 "bounds": [0, 10],
#                 "initial": 6,
#                 "step": 1,
#                 "log": False,
#             },
#         ]
#     )

#     assert param.bounds == {"x1": [0, 10], "x2": [0, 10]}
#     assert param.initial == {"x1": 3, "x2": 6}
#     assert param.step == {"x1": 1, "x2": 1}
#     assert param.log == {"x1": False, "x2": False}
#     assert param.type == {"x1": "float", "x2": "float"}
#     assert param.names == ["x1", "x2"]

#     assert param.values["x1"] == 3
#     param.values["x1"] = 9
#     assert param.values["x1"] == 9
