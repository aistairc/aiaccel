"""Fit regression models from collected modelbridge train pairs."""

from __future__ import annotations

from numpy.typing import NDArray
from typing import Any

import argparse
from collections.abc import Mapping, Sequence
import csv
import json
import os
from pathlib import Path
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import yaml

SUPPORTED_REGRESSION_KINDS = {"linear", "polynomial", "gpr"}
DEFAULT_REGRESSION_KIND = "linear"
DEFAULT_POLYNOMIAL_DEGREE = 2
FloatArray = NDArray[np.float64]


class GPyMultiOutputRegressor:
    """Pickle-friendly wrapper that provides multi-target prediction for GPy models."""

    def __init__(self, models: list[Any]) -> None:
        """Initialize the wrapper with per-target GPy regression models.

        Args:
            models: Per-target GPy model instances.
        """
        self.models = models

    def predict(self, features: Any) -> FloatArray:
        """Run inference for all target models and return a 2D prediction matrix.

        Args:
            features: Feature matrix with shape ``(n_samples, n_features)``.

        Returns:
            Prediction matrix with shape ``(n_samples, n_targets)``.
        """
        feature_array = np.asarray(features, dtype=float)
        if not self.models:
            return np.empty((feature_array.shape[0], 0), dtype=float)
        columns = [np.asarray(model.predict(feature_array)[0], dtype=float).reshape(-1) for model in self.models]
        return np.column_stack(columns)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the fit step.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Modelbridge Fit Step")
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace directory")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config.yaml")
    return parser.parse_args(argv)


def _as_int(value: Any, *, key: str, min_value: int) -> int:
    """Parse an integer and validate lower bound.

    Args:
        value: Raw value to parse.
        key: Config key name used in the error message.
        min_value: Minimum accepted value.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the value is lower than ``min_value`` or not integer-like.
    """
    parsed = int(value)
    if parsed < min_value:
        raise ValueError(f"{key} must be >= {min_value}")
    return parsed


def _resolve_config_path(workspace: Path, *, config_path: Path | None = None) -> Path | None:
    """Resolve config path for regression settings.

    Resolution order:
    1. Explicit ``config_path`` argument.
    2. ``MODELBRIDGE_CONFIG_FILE`` environment variable.
    3. ``CONFIG_FILE`` environment variable.
    4. ``workspace/config.yaml``.
    5. ``workspace/../config/config.yaml`` (example layout).

    Args:
        workspace: Workspace root path.
        config_path: Optional explicit config path.

    Returns:
        Resolved existing config path, or ``None`` when no candidate exists.

    Raises:
        FileNotFoundError: If ``config_path`` is explicitly provided but does not exist.
    """
    if config_path is not None:
        resolved = config_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found at {resolved}")
        return resolved

    env_candidates = [os.environ.get("MODELBRIDGE_CONFIG_FILE"), os.environ.get("CONFIG_FILE")]
    path_candidates: list[Path] = []
    for raw in env_candidates:
        if raw:
            candidate = Path(raw).expanduser()
            candidate = (Path.cwd() / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            path_candidates.append(candidate)
    path_candidates.extend((workspace / "config.yaml", workspace.parent / "config" / "config.yaml"))
    for candidate in path_candidates:
        if candidate.exists():
            return candidate
    return None


def _load_regression_settings(workspace: Path, *, config_path: Path | None = None) -> dict[str, Any]:
    """Load regression settings from config file with safe defaults.

    Args:
        workspace: Workspace root path.
        config_path: Optional explicit config path.

    Returns:
        Normalized regression settings dictionary.

    Raises:
        FileNotFoundError: If explicit config path is provided and missing.
        ValueError: If config schema or parameter values are invalid.
    """
    settings: dict[str, Any] = {
        "kind": DEFAULT_REGRESSION_KIND,
        "degree": DEFAULT_POLYNOMIAL_DEGREE,
        "noise": None,
        "config_path": None,
    }
    resolved_path = _resolve_config_path(workspace, config_path=config_path)
    if resolved_path is None:
        return settings

    loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    root_config: Mapping[str, Any] = loaded if isinstance(loaded, Mapping) else {}
    raw_regression = root_config.get("regression", {})
    if raw_regression is None:
        raw_regression = {}
    if not isinstance(raw_regression, Mapping):
        raise ValueError("regression must be a mapping")

    raw_kind = raw_regression.get("kind", root_config.get("regression_kind", DEFAULT_REGRESSION_KIND))
    kind = str(raw_kind).strip().lower()
    if kind not in SUPPORTED_REGRESSION_KINDS:
        supported = ", ".join(sorted(SUPPORTED_REGRESSION_KINDS))
        raise ValueError(f"Unsupported regression kind: {kind}. Supported: {supported}")

    degree = _as_int(
        raw_regression.get("degree", root_config.get("regression_degree", DEFAULT_POLYNOMIAL_DEGREE)),
        key="regression.degree",
        min_value=1,
    )

    noise_raw = raw_regression.get("noise", root_config.get("gpr_noise"))
    noise: float | None
    if noise_raw is None:
        noise = None
    else:
        noise = float(noise_raw)
        if noise <= 0:
            raise ValueError("regression.noise must be > 0")

    settings.update(
        {
            "kind": kind,
            "degree": degree,
            "noise": noise,
            "config_path": str(resolved_path),
        }
    )
    return settings


def _load_gpy_module() -> Any:
    """Import and return the GPy module.

    Returns:
        Imported GPy module.

    Raises:
        RuntimeError: If GPy is not installed.
    """
    try:
        import GPy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional dependency installation.
        raise RuntimeError(
            "GPy is required when regression.kind is 'gpr'. Install with 'pip install .[modelbridge]'."
        ) from exc
    return GPy


def _fit_gpr_model(features: FloatArray, targets: FloatArray, *, noise: float | None) -> GPyMultiOutputRegressor:
    """Fit per-target GPy regressors and bundle them.

    Args:
        features: Feature matrix.
        targets: Target matrix.
        noise: Optional fixed noise variance.

    Returns:
        Wrapped multi-output GPy regressor.
    """
    gpy = _load_gpy_module()
    input_dim = features.shape[1]
    models: list[Any] = []
    for target_index in range(targets.shape[1]):
        kernel = gpy.kern.RBF(input_dim=input_dim)
        kwargs: dict[str, Any] = {}
        if noise is not None:
            kwargs["noise_var"] = noise
        model = gpy.models.GPRegression(features, targets[:, target_index : target_index + 1], kernel, **kwargs)
        models.append(model)
    return GPyMultiOutputRegressor(models=models)


def _train_regression_model(
    *,
    features: FloatArray,
    targets: FloatArray,
    regression_settings: Mapping[str, Any],
) -> tuple[Any, str]:
    """Train selected regression model.

    Args:
        features: Feature matrix.
        targets: Target matrix.
        regression_settings: Normalized regression settings.

    Returns:
        Tuple of trained model and model type label.
    """
    kind = str(regression_settings["kind"])
    if kind == "linear":
        model = LinearRegression()
        model.fit(features, targets)
        return model, "LinearRegression"

    if kind == "polynomial":
        degree = int(regression_settings["degree"])
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        model.fit(features, targets)
        return model, "PolynomialRegression"

    noise = regression_settings["noise"]
    model = _fit_gpr_model(features, targets, noise=noise)
    return model, "GPyGaussianProcessRegression"


def run_fit_model(workspace: Path, *, config_path: Path | None = None) -> Path | None:
    """Fit regression model from train pairs and persist model artifacts.

    Args:
        workspace: Workspace directory containing ``pairs/train_pairs.csv``.
        config_path: Optional path to workflow config YAML.

    Returns:
        Path to the serialized model on success, or ``None`` when the training CSV is empty.

    Raises:
        FileNotFoundError: If training CSV is missing or explicit config path does not exist.
        ValueError: If required CSV columns or regression settings are invalid.
        RuntimeError: If ``regression.kind='gpr'`` is selected but GPy is unavailable.
    """
    train_csv = workspace / "pairs" / "train_pairs.csv"
    model_dir = workspace / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training data not found at {train_csv}")

    records: list[dict[str, str]] = []
    with train_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records.extend(reader)
    if not records:
        print("Warning: Training CSV is empty. Model fitting skipped.")
        return None

    macro_keys = sorted(key for key in records[0] if key.startswith("macro_"))
    micro_keys = sorted(key for key in records[0] if key.startswith("micro_"))
    if not macro_keys or not micro_keys:
        raise ValueError("CSV must contain both macro_ and micro_ prefixed columns.")

    features = np.asarray([[float(record[key]) for key in macro_keys] for record in records], dtype=float)
    targets = np.asarray([[float(record[key]) for key in micro_keys] for record in records], dtype=float)
    regression_settings = _load_regression_settings(workspace, config_path=config_path)

    model, model_type = _train_regression_model(
        features=features,
        targets=targets,
        regression_settings=regression_settings,
    )
    print(
        f"Fitting model with {len(records)} samples "
        f"(Features: {len(macro_keys)}, Targets: {len(micro_keys)}, Kind: {regression_settings['kind']})"
    )

    model_path = model_dir / "regression_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    meta_path = model_dir / "model_meta.json"
    meta_payload: dict[str, Any] = {
        "macro_features": macro_keys,
        "micro_targets": micro_keys,
        "n_samples": len(records),
        "model_type": model_type,
        "regression_kind": regression_settings["kind"],
    }
    if regression_settings["kind"] == "polynomial":
        meta_payload["regression_degree"] = int(regression_settings["degree"])
    if regression_settings["kind"] == "gpr" and regression_settings["noise"] is not None:
        meta_payload["gpr_noise"] = float(regression_settings["noise"])
    if regression_settings["config_path"] is not None:
        meta_payload["config_path"] = regression_settings["config_path"]

    meta_path.write_text(
        json.dumps(meta_payload, indent=2),
        encoding="utf-8",
    )
    print(f"[Fit] Regression model ({model_type}) saved to {model_path}")
    return model_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fit-model step CLI.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Process exit code. ``0`` on success, ``1`` on input/runtime errors.
    """
    args = parse_args(argv)
    try:
        config_path = Path(args.config) if args.config else None
        run_fit_model(Path(args.workspace), config_path=config_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
