import csv
import glob
import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from aiaccel.hpo.samplers.nelder_mead_sampler import NelderMeadEmptyError, NelderMeadSampler


# https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda#ackley-function
def ackley(x: list[float]) -> float:
    # Ackley function
    x = np.asarray(x)
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(x) * np.sum(x**2)))
    t3 = np.e
    t4 = -np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
    return float(t1 + t2 + t3 + t4)


search_space = {
    "x1": (-32.768, 32.768),
    "x2": (-32.768, 32.768),
    "x3": (-32.768, 32.768),
    "x4": (-32.768, 32.768),
    "x5": (-32.768, 32.768),
    "x6": (-32.768, 32.768),
    "x7": (-32.768, 32.768),
    "x8": (-32.768, 32.768),
    "x9": (-32.768, 32.768),
    "x10": (-32.768, 32.768),
}


def optimize(
    study: optuna.Study,
    func: Callable[[list[float]], float],
    result_csv_name: str,
    search_space: dict[str, tuple[float, float]] = search_space,
    num_trial: int = 1000,
    num_parallel: int = 10,
    num_tell_trial: int = 1,
) -> None:
    trials: list[optuna.trial.Trial] = []
    params = []
    p = Pool(num_parallel)
    results = []
    csv_array: list[list[str | float]] = [["step", "value"]]

    finished_trials = []
    finished_results = []

    for step in range(int(num_trial / num_parallel)):
        while len(trials) < num_parallel:
            try:
                trial = study.ask()
            except NelderMeadEmptyError:
                break
            param = []
            for name, distribution in search_space.items():
                param.append(trial.suggest_float(name, *distribution))
            trials.append(trial)
            params.append(param)

        try:
            results = p.map(func, params)

        except Exception as e:
            print(e)

        for obj in results:
            csv_array.append([step, obj])

        finished_trials += trials
        finished_results += results

        if num_tell_trial <= len(finished_trials):
            for trial, obj in zip(finished_trials, finished_results, strict=False):
                frozentrial = study.tell(trial, obj)
                study._log_completed_trial(frozentrial)

            finished_trials = []
            finished_results = []

        trials = []
        params = []
        results = []

    with open(result_csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_array)


def get_min_values(values: list[float]) -> list[float]:
    min_values = []
    min_value = np.inf
    for value in values:
        if value < min_value:
            min_value = value
        min_values.append(min_value)

    return min_values


def plot(file_names: list[str], png_name: str, title: str) -> None:
    label_names = ["nelder_mead", "TPE", "nelder_mead+TPE"]
    colors = ["r", "g", "b"]

    plt.figure(figsize=(12, 8))

    for file_name, color, label in zip(file_names, colors, label_names, strict=False):
        df = pd.read_csv(file_name)

        step = df["step"]
        values = df["value"]

        plt.plot(step, values, marker=".", linewidth=0, color=color, alpha=0.2)

        min_values = get_min_values(list(values))
        plt.plot(step, min_values, color=color, label=label)

    plt.title(title, fontsize=18)
    plt.xlabel("Parallel Step", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.legend()

    plt.savefig(png_name)

    plt.clf()
    plt.close()


def plot_mean_std(file_patterns: list[str], png_name: str, title: str) -> None:
    label_names = ["nelder_mead", "TPE", "nelder_mead+TPE"]
    colors = ["r", "g", "b"]

    plt.figure(figsize=(12, 8))

    for file_pattern, color, label in zip(file_patterns, colors, label_names, strict=False):
        file_names = glob.glob(file_pattern)

        min_values_array = []

        df = pd.read_csv(file_names[0])
        df_min = df.groupby("step").min()
        step = df_min.index

        for file_path in file_names:
            df = pd.read_csv(file_path)
            min_values = get_min_values([item[0] for item in df_min.values])
            min_values_array.append(min_values)

        min_values_mean = np.array(min_values_array).mean(axis=0)
        min_values_std = np.array(min_values_array).std(axis=0)

        plt.plot(step, min_values_mean, color=color, label=label)
        plt.fill_between(
            df_min.index, min_values_mean + min_values_std, min_values_mean - min_values_std, alpha=0.2, color=color
        )

    plt.title(title, fontsize=18)
    plt.xlabel("Parallel Step", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.legend()

    plt.savefig(png_name)

    plt.clf()
    plt.close()


def compare_optimizer(num_trial: int, num_parallel: int, func: Callable[[list[float]], float], dir_name: str) -> None:
    os.makedirs(dir_name, exist_ok=True)

    for seed in range(42, 52):
        result_csv_names = [
            f"{dir_name}/result_nelder_mead_{dir_name}_seed{seed}_{num_trial}trial.csv",
            f"{dir_name}/result_TPE_{dir_name}_seed{seed}_{num_trial}trial.csv",
            f"{dir_name}/result_nelder_mead_TPE_{dir_name}_seed{seed}_{num_trial}trial.csv",
        ]

        study = optuna.create_study(sampler=NelderMeadSampler(search_space=search_space, seed=seed))
        optimize(study, func, result_csv_names[0], search_space, num_trial, num_parallel)

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
        optimize(study, func, result_csv_names[1], search_space, num_trial, num_parallel)

        study = optuna.create_study(
            sampler=NelderMeadSampler(
                search_space=search_space, seed=seed, block=False, sub_sampler=optuna.samplers.TPESampler(seed=seed)
            )
        )
        optimize(study, func, result_csv_names[2], search_space, num_trial, num_parallel, num_tell_trial=10)

        plot(result_csv_names, f"{dir_name}/{dir_name}_seed{seed}_{num_trial}trial.png", f"{dir_name}")

    result_csv_patterns = [
        f"{dir_name}/result_nelder_mead_{dir_name}_seed*_{num_trial}trial.csv",
        f"{dir_name}/result_TPE_{dir_name}_seed*_{num_trial}trial.csv",
        f"{dir_name}/result_nelder_mead_TPE_{dir_name}_seed*_{num_trial}trial.csv",
    ]

    plot_mean_std(result_csv_patterns, f"{dir_name}/{dir_name}_avg_{num_trial}trial.png", f"{dir_name}")


if __name__ == "__main__":
    # ackley_100step_10parallel
    compare_optimizer(num_trial=1000, num_parallel=10, func=ackley, dir_name="ackley_100step_10parallel")

    # # ackley_1000step_series
    compare_optimizer(num_trial=1000, num_parallel=1, func=ackley, dir_name="ackley_1000step_series")
