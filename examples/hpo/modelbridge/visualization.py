"""Visualization utilities for model bridging results."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Type aliases
FilePath = str | Path
NumPyArray = np.ndarray


class Visualizer:
    """Visualization utility for model bridging results."""

    def __init__(self, figsize: tuple = (12, 8), dpi: int = 100):
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
            ax.scatter(
                macro_params[:, macro_param_idx], micro_params[:, i], alpha=0.6, s=30
            )
            ax.set_xlabel(f"{macro_param_names[macro_param_idx]}")
            ax.set_ylabel(f"{micro_param_names[i]}")
            ax.set_title(
                f"{macro_param_names[macro_param_idx]} vs {micro_param_names[i]}"
            )
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
                correlation = np.corrcoef(true_values[:, i], predicted_values[:, i])[
                    0, 1
                ]
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
                    continue  # Skip non-numeric
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize modelbridge results")
    parser.add_argument("scenario_dir", type=Path, help="Path to the scenario directory (e.g. work/modelbridge/simple/simple)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory for plots")
    args = parser.parse_args()

    scenario_dir = args.scenario_dir
    output_dir = args.output or scenario_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = Visualizer()

    # 1. Plot Parameter Relationships (Train Pairs)
    train_file = scenario_dir / "train_pairs.csv"
    if train_file.exists():
        print(f"Loading {train_file}...")
        data = load_csv_data(train_file)
        
        macro_keys = [k for k in data.keys() if k.startswith("macro_")]
        micro_keys = [k for k in data.keys() if k.startswith("micro_")]
        
        if macro_keys and micro_keys:
            macro_data = np.column_stack([data[k] for k in macro_keys])
            micro_data = np.column_stack([data[k] for k in micro_keys])
            
            viz.plot_parameter_relationship(
                macro_data, 
                micro_data, 
                [k.replace("macro_", "") for k in macro_keys],
                [k.replace("micro_", "") for k in micro_keys],
                output_dir=output_dir
            )
    else:
        print(f"Warning: {train_file} not found.")

    # 2. Plot Prediction Accuracy (Test Predictions)
    test_file = scenario_dir / "test_predictions.csv"
    if test_file.exists():
        print(f"Loading {test_file}...")
        data = load_csv_data(test_file)
        
        actual_keys = [k for k in data.keys() if k.startswith("actual_")]
        pred_keys = [k for k in data.keys() if k.startswith("pred_")]
        
        # Sort to match pairs
        actual_keys.sort()
        pred_keys.sort()
        
        if actual_keys and pred_keys and len(actual_keys) == len(pred_keys):
            actual_data = np.column_stack([data[k] for k in actual_keys])
            pred_data = np.column_stack([data[k] for k in pred_keys])
            
            param_names = [k.replace("actual_", "") for k in actual_keys]
            
            viz.plot_prediction_accuracy(
                actual_data,
                pred_data,
                param_names,
                output_dir=output_dir
            )
    else:
        print(f"Warning: {test_file} not found.")


if __name__ == "__main__":
    main()