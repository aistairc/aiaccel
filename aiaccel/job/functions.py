from __future__ import annotations


def param_to_args_key_value(param: dict) -> str:
    """
    Example:
    param = {
        'x': 0.5,
        'y': 0.3,
        ...
    }
    return "-x=0.5 -y=0.3 ..."
    """
    return " ".join([f"--{k}={v}" for k, v in param.items()])


def param_to_args_value(param: dict) -> str:
    """
    Example:
    param = {
        'x': 0.5,
        'y': 0.3,
        ...
    }
    return "0.5 0.3 ..."
    """
    return " ".join([f"{v}" for k, v in param.items()])
