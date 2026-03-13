"""Evaluate the fitted modelbridge regression model."""

from __future__ import annotations

from typing import Any

import argparse
from collections.abc import Sequence
import csv
import json
from pathlib import Path
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the evaluate step.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Modelbridge Evaluate Step")
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace directory")
    return parser.parse_args(argv)


def run_evaluate(workspace: Path) -> Path | None:
    """Evaluate the regression model against test pairs and export artifacts.

    Args:
        workspace: Workspace directory containing model artifacts and test pair CSV.

    Returns:
        Path to the evaluation summary JSON on success, or ``None`` when the test CSV is empty.

    Raises:
        FileNotFoundError: If required test data or model artifacts are missing.
    """
    test_csv = workspace / "pairs" / "test_pairs.csv"
    model_path = workspace / "models" / "regression_model.pkl"
    meta_path = workspace / "models" / "model_meta.json"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test data not found at {test_csv}")
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model or metadata not found in {workspace / 'models'}")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)  # noqa: S301 - trusted local artifact.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    macro_keys = list(meta["macro_features"])
    micro_keys = list(meta["micro_targets"])

    records: list[dict[str, str]] = []
    with test_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records.extend(reader)
    if not records:
        print("Warning: Test CSV is empty. Evaluation skipped.")
        return None

    features: list[list[float]] = []
    targets: list[list[float]] = []
    run_ids: list[str] = []
    for record in records:
        run_ids.append(record.get("run_id", "unknown"))
        features.append([float(record.get(key, 0.0)) for key in macro_keys])
        targets.append([float(record.get(key, 0.0)) for key in micro_keys])
    predictions = model.predict(features)

    mse = float(mean_squared_error(targets, predictions))
    mae = float(mean_absolute_error(targets, predictions))
    r2 = float(r2_score(targets, predictions))
    print(f"Evaluation Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    pred_csv = workspace / "pairs" / "test_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["run_id", *[f"true_{key}" for key in micro_keys], *[f"pred_{key}" for key in micro_keys]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, run_id in enumerate(run_ids):
            row: dict[str, Any] = {"run_id": run_id}
            for target_index, key in enumerate(micro_keys):
                row[f"true_{key}"] = float(targets[index][target_index])
                row[f"pred_{key}"] = float(predictions[index][target_index])
            writer.writerow(row)

    summary_path = workspace / "models" / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metrics": {"mse": mse, "mae": mae, "r2": r2},
                "n_test_samples": len(records),
                "model_type": meta["model_type"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Evaluate] Summary saved to {summary_path}")
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluate step CLI.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Process exit code. ``0`` on success, ``1`` on input/runtime errors.
    """
    args = parse_args(argv)
    try:
        run_evaluate(Path(args.workspace))
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
