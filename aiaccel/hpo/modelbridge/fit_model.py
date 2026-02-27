"""Fit regression model step."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
from pathlib import Path
import pickle

from sklearn.linear_model import LinearRegression


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Modelbridge Fit Step")
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace directory")
    return parser.parse_args(argv)


def run_fit_model(workspace: Path) -> Path | None:
    """Fit linear regression from train_pairs.csv and persist model artifacts."""
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

    features = [[float(record[key]) for key in macro_keys] for record in records]
    targets = [[float(record[key]) for key in micro_keys] for record in records]

    print(f"Fitting model with {len(records)} samples (Features: {len(macro_keys)}, Targets: {len(micro_keys)})")
    model = LinearRegression()
    model.fit(features, targets)

    model_path = model_dir / "regression_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    meta_path = model_dir / "model_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "macro_features": macro_keys,
                "micro_targets": micro_keys,
                "n_samples": len(records),
                "model_type": "LinearRegression",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Fit] Regression model saved to {model_path}")
    return model_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for fit-model step."""
    args = parse_args(argv)
    try:
        run_fit_model(Path(args.workspace))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
