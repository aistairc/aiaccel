"""Visualization utilities for model bridging results."""

from numpy.typing import NDArray
from typing import Any

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import optuna
    from sklearn.preprocessing import PolynomialFeatures

    _HAS_OPTUNA_SKLEARN = True
except ImportError:
    _HAS_OPTUNA_SKLEARN = False

# Type aliases
FilePath = str | Path
NumPyArray = NDArray[Any]


class Visualizer:
    """Visualization utility for model bridging results."""

    def __init__(self, figsize: tuple[int, int] = (12, 8), dpi: int = 100):
        """Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: Default DPI for plots

        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use("default")

    def plot_parameter_relationship(
        self,
        macro_params: NumPyArray,
        micro_params: NumPyArray,
        macro_param_names: list[str],
        micro_param_names: list[str],
        title: str = "Parameter Relationships",
        output_dir: FilePath | None = None,
        filename: str = "parameter_relationships.png",
    ) -> None:
        """Plot relationships between macro and micro parameters."""
        if macro_params.size == 0 or micro_params.size == 0:
            print("No data to plot parameter relationships.")
            return

        n_micro_params = micro_params.shape[1]
        n_cols = min(3, n_micro_params)
        n_rows = (n_micro_params + n_cols - 1) // n_cols

        _fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figsize[0], self.figsize[1] * n_rows / 2),
            dpi=self.dpi,
        )

        if n_micro_params == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        macro_param_idx = 0  # Use first macro parameter for x-axis

        for i in range(n_micro_params):
            ax = axes[i]
            ax.scatter(macro_params[:, macro_param_idx], micro_params[:, i], alpha=0.6, s=30)
            ax.set_xlabel(f"{macro_param_names[macro_param_idx]}")
            ax.set_ylabel(f"{micro_param_names[i]}")
            ax.set_title(f"{macro_param_names[macro_param_idx]} vs {micro_param_names[i]}")
            ax.grid(True, alpha=0.3)

        for i in range(n_micro_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
        plt.close()

    def plot_prediction_accuracy(
        self,
        true_values: NumPyArray,
        predicted_values: NumPyArray,
        param_names: list[str],
        title: str = "Prediction Accuracy",
        output_dir: FilePath | None = None,
        filename: str = "prediction_accuracy.png",
    ) -> None:
        """Plot prediction accuracy for each parameter."""
        if true_values.size == 0 or predicted_values.size == 0:
            print("No data to plot prediction accuracy.")
            return

        n_params = true_values.shape[1]
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        _fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.figsize[0], self.figsize[1] * n_rows / 2),
            dpi=self.dpi,
        )

        if n_params == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(n_params):
            ax = axes[i]
            ax.scatter(
                predicted_values[:, i],
                true_values[:, i],
                alpha=0.6,
                s=30,
                label="Predictions",
            )

            min_val = min(true_values[:, i].min(), predicted_values[:, i].min())
            max_val = max(true_values[:, i].max(), predicted_values[:, i].max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                alpha=0.8,
                label="Perfect",
            )

            ax.set_xlabel(f"Predicted {param_names[i]}")
            ax.set_ylabel(f"True {param_names[i]}")
            ax.set_title(f"{param_names[i]} Prediction")
            ax.grid(True, alpha=0.3)
            ax.legend()

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                correlation = np.corrcoef(true_values[:, i], predicted_values[:, i])[0, 1]
                r_squared = correlation**2 if not np.isnan(correlation) else 0.0
            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
        plt.close()


def load_csv_data(filepath: Path) -> dict[str, list[float]]:
    """Load CSV data into a dictionary of columns."""
    data: dict[str, list[float]] = {}
    with filepath.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except ValueError:
                    continue
    return data


def scan_optuna_studies(runs_dir: Path) -> list[tuple[int, dict[str, float], dict[str, float]]]:
    """Scan directory for Optuna DBs and extract best params."""
    if not _HAS_OPTUNA_SKLEARN:
        print("Optuna or Scikit-learn not available. Skipping DB scan.")
        return []

    samples = []
    if not runs_dir.exists():
        return []

    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        try:
            run_idx = int(run_path.name)
        except ValueError:
            continue

        macro_db = run_path / "macro" / "optuna.db"
        micro_db = run_path / "micro" / "optuna.db"

        if not macro_db.exists() or not micro_db.exists():
            continue

        macro_best = _load_best_param(macro_db)
        micro_best = _load_best_param(micro_db)

        if macro_best and micro_best:
            samples.append((run_idx, macro_best, micro_best))

    return samples


def _load_best_param(db_path: Path) -> dict[str, float] | None:
    storage = f"sqlite:///{db_path.resolve()}"
    try:
        # We don't know the exact study name easily without parsing directory or trial error.
        # But aiaccel-hpo usually creates one study per DB in this structure.
        # We can try to list studies.
        studies = optuna.study.get_all_study_summaries(storage=storage)
        if not studies:
            return None
        study = optuna.load_study(study_name=studies[0].study_name, storage=storage)
        if study.best_trial:
            return {k: float(v) for k, v in study.best_trial.params.items()}
    except Exception:
        return None
    return None


def predict_from_model(model_path: Path, features_list: list[dict[str, float]]) -> list[dict[str, float]] | None:
    """Predict using the saved regression model."""
    if not model_path.exists() or not _HAS_OPTUNA_SKLEARN:
        return None

    try:
        with model_path.open("r") as f:
            model_dict = json.load(f)

        feature_names = model_dict["feature_names"]
        target_names = model_dict["target_names"]
        kind = model_dict["kind"]

        x_data = np.asarray([[f[k] for k in feature_names] for f in features_list])

        if kind in ["linear", "polynomial"]:
            degree = model_dict["degree"]
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly.fit_transform(x_data)

            coef = np.asarray(model_dict["coefficients"])
            intercept = np.asarray(model_dict["intercept"])

            y_pred = x_poly @ coef.T + intercept

        elif kind == "gpr":
            # GPR loading requires GPy which might not be installed or complex to mock here.
            # We skip GPR fallback for now or need GPy.
            print("GPR prediction fallback not fully supported in visualization script.")
            return None
        else:
            return None

        results = []
        for row in y_pred:
            results.append({k: float(v) for k, v in zip(target_names, row, strict=True)})
        return results

    except Exception as e:
        print(f"Failed to predict from model: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize modelbridge results")
    parser.add_argument("scenario_dir", type=Path, help="Path to the scenario directory")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory for plots")
    args = parser.parse_args()

    scenario_dir = args.scenario_dir
    output_dir = args.output or scenario_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = Visualizer()

    # --- 1. Parameter Relationships ---
    macro_keys, micro_keys = [], []
    macro_data, micro_data = None, None

    train_file = scenario_dir / "train_pairs.csv"
    if train_file.exists():
        print(f"Loading {train_file}...")
        data = load_csv_data(train_file)
        macro_keys = sorted([k for k in data if k.startswith("macro_")])
        micro_keys = sorted([k for k in data if k.startswith("micro_")])

        if macro_keys and micro_keys:
            macro_data = np.column_stack([data[k] for k in macro_keys])
            micro_data = np.column_stack([data[k] for k in micro_keys])
    else:
        print(f"Warning: {train_file} not found. Attempting to scan Optuna DBs...")
        samples = scan_optuna_studies(scenario_dir / "runs" / "train")
        if samples:
            print(f"Found {len(samples)} samples from DB.")
            # Assume keys from first sample
            first_macro = samples[0][1]
            first_micro = samples[0][2]
            macro_keys = sorted([f"macro_{k}" for k in first_macro])
            micro_keys = sorted([f"micro_{k}" for k in first_micro])

            m_list = [[s[1][k.replace("macro_", "")] for k in macro_keys] for s in samples]
            u_list = [[s[2][k.replace("micro_", "")] for k in micro_keys] for s in samples]

            macro_data = np.array(m_list)
            micro_data = np.array(u_list)

    if macro_data is not None and micro_data is not None:
        viz.plot_parameter_relationship(
            macro_data,
            micro_data,
            [k.replace("macro_", "") for k in macro_keys],
            [k.replace("micro_", "") for k in micro_keys],
            output_dir=output_dir,
        )
    else:
        print("Could not load training data.")

    # --- 2. Prediction Accuracy ---
    test_file = scenario_dir / "test_predictions.csv"
    actual_data, pred_data = None, None
    param_names = []

    if test_file.exists():
        print(f"Loading {test_file}...")
        data = load_csv_data(test_file)
        actual_keys = sorted([k for k in data if k.startswith("actual_")])
        pred_keys = sorted([k for k in data if k.startswith("pred_")])

        if actual_keys and pred_keys and len(actual_keys) == len(pred_keys):
            actual_data = np.column_stack([data[k] for k in actual_keys])
            pred_data = np.column_stack([data[k] for k in pred_keys])
            param_names = [k.replace("actual_", "") for k in actual_keys]
    else:
        print(f"Warning: {test_file} not found. Attempting to generate predictions...")
        # Load Eval DBs
        eval_samples = scan_optuna_studies(scenario_dir / "runs" / "eval")
        model_path = scenario_dir / "models" / "regression_model.json"

        if eval_samples and model_path.exists():
            features = [s[1] for s in eval_samples]
            targets = [s[2] for s in eval_samples]

            preds = predict_from_model(model_path, features)

            if preds:
                # Prepare arrays
                # Sort keys to match
                t_keys = sorted(targets[0].keys())
                p_keys = sorted(preds[0].keys())  # Should match t_keys

                actual_data = np.array([[t[k] for k in t_keys] for t in targets])
                pred_data = np.array([[p[k] for k in p_keys] for p in preds])
                param_names = t_keys
                print(f"Generated predictions for {len(eval_samples)} eval samples.")

    if actual_data is not None and pred_data is not None:
        viz.plot_prediction_accuracy(actual_data, pred_data, param_names, output_dir=output_dir)
    else:
        print("Could not load/generate prediction data.")


if __name__ == "__main__":
    main()
