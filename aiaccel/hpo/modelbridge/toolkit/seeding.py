"""Seed policy helpers."""

from __future__ import annotations

from ..config import SeedPolicyConfig, SeedUserValues
from ..layout import Role, Target

RUN_ID_OFFSET_STRIDE = 100000


def resolve_seed(
    policy: SeedPolicyConfig,
    *,
    role: Role,
    target: Target,
    run_id: int,
    fallback_base: int,
) -> int:
    """Resolve a deterministic seed from one policy.

    Args:
        policy: Seed policy definition.
        role: Role (`train` or `eval`).
        target: Target (`macro` or `micro`).
        run_id: Zero-based run index.
        fallback_base: Base seed used when policy base is omitted.

    Returns:
        int: Resolved seed value.

    Raises:
        ValueError: If user-defined policy does not provide a value for run index.
    """
    if policy.mode == "auto_increment":
        base = fallback_base if policy.base is None else policy.base
        return base + _group_index(role, target) * RUN_ID_OFFSET_STRIDE + run_id

    if policy.user_values is None:
        raise ValueError("user_values must be provided when mode=user_defined")

    values = _select_user_values(policy.user_values, role, target)
    if run_id < 0 or run_id >= len(values):
        raise ValueError(f"Missing user-defined seed for {role}/{target} run_id={run_id}")
    return values[run_id]


def _group_index(role: Role, target: Target) -> int:
    """Return deterministic seed group index by role/target."""
    if role == "train" and target == "macro":
        return 0
    if role == "train" and target == "micro":
        return 1
    if role == "eval" and target == "macro":
        return 2
    return 3


def _select_user_values(values: SeedUserValues, role: Role, target: Target) -> list[int]:
    """Select seed list for one role/target pair."""
    if role == "train" and target == "macro":
        return values.train_macro
    if role == "train" and target == "micro":
        return values.train_micro
    if role == "eval" and target == "macro":
        return values.eval_macro
    return values.eval_micro
