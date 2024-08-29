import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes


def get_min_values(values: list[float]) -> list[float]:
    min_values = []
    min_value = np.inf
    for value in values:
        if value < min_value:
            min_value = value
        min_values.append(min_value)

    return min_values


def plot_dim_vs_min_value(ax: Axes, csv_names_for_dim: list[list[list[str]]], title: str) -> None:
    label_names = ["nm", "TPE-mcT-mvF", "nm+subTPE-mcT-mvF"]
    colors = ["r", "g", "b"]

    dim = ["2", "3", "5", "10", "20", "40"]

    # dim
    min_values_for_dim = []
    for csv_names_for_algorism in csv_names_for_dim:
        # nm, tpe, nm_subtpe
        min_values_for_algorism = []
        for csv_names in csv_names_for_algorism:
            min_values = []
            for csv_name in csv_names:
                df = pd.read_csv(csv_name)

                df_values = df["value - f_opt"]
                print(df_values)
                print(min(df_values))
                min_values.append(min(df_values))
            min_values_for_algorism.append(min_values)
        min_values_for_dim.append(min_values_for_algorism)

    print(len(min_values_for_dim))

    for i in range(3):
        values = [item[i] for item in min_values_for_dim]
        print(values)
        values_mean = np.array(values).mean(axis=1)
        values_std = np.array(values).std(axis=1)
        ax.errorbar(
            dim,
            values_mean,
            yerr=values_std,
            capsize=5,
            markersize=10,
            ecolor=colors[i],
            markeredgecolor=colors[i],
            color=colors[i],
            label=label_names[i],
        )

    if title == "f6":
        ax.set_ylim(-1000, 10000)
    ax.set_title(title)
    ax.grid(axis="both")
    ax.legend(fontsize=6)


def compare_optimizer(base_dir: str = ".") -> None:
    fig, ax = plt.subplots(5, 5, figsize=(16, 20))
    if isinstance(ax, Axes):
        return

    for num_of_f in range(1, 25):
        result_csv_list_for_dim = []
        for num_of_dm in [2, 3, 5, 10, 20, 40]:
            result_csv_patterns = [
                f"{base_dir}/nelder-mead/optuna_csv/optuna-nelder-mead-func_id{num_of_f}-dim{num_of_dm}-instance*/f{num_of_f}/DM{num_of_dm:02}/result_bbob_f{num_of_f:03}_i*_d{num_of_dm:02}_*_fopt.csv",
                f"{base_dir}/TPE/optuna_csv/optuna-TPE-func_id{num_of_f}-dim{num_of_dm}-instance*/f{num_of_f}/DM{num_of_dm:02}/result_bbob_f{num_of_f:03}_i*_d{num_of_dm:02}_*_fopt.csv",
                f"{base_dir}/nelder-mead-subTPE/optuna_csv/optuna-nelder-mead-subTPE-func_id{num_of_f}-dim{num_of_dm}-instance*/f{num_of_f}/DM{num_of_dm:02}/result_bbob_f{num_of_f:03}_i*_d{num_of_dm:02}_*_fopt.csv",
            ]

            result_csv_list = [sorted(glob.glob(pattern)) for pattern in result_csv_patterns]
            print(result_csv_list)
            result_csv_list_for_dim.append(result_csv_list)

        plot_dim_vs_min_value(
            ax[int((num_of_f - 1) / 5), int((num_of_f - 1) % 5)], result_csv_list_for_dim, f"f{num_of_f}"
        )

    plt.savefig("result_bbob_dim_vs_value-fopt_parallel.png")
    plt.show()


if __name__ == "__main__":
    compare_optimizer()
