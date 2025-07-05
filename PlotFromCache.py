import os
import pickle
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_success_failure_subplots(stats_by_iteration: dict):
    num_iterations = len(stats_by_iteration)
    if num_iterations == 0:
        print("No data to plot.")
        return

    cols = 2
    rows = (num_iterations + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, (iteration, stat_list) in enumerate(sorted(stats_by_iteration.items())):
        ax = axes[idx]
        method_success = defaultdict(int)
        method_failure = defaultdict(int)

        for stat in stat_list:
            method = stat.get_method_name()
            method_success[method] += stat.get_n_mission_success()
            method_failure[method] += stat.get_n_mission_failed()

        methods = sorted(set(method_success.keys()).union(method_failure.keys()))
        successes = [method_success[m] for m in methods]
        failures = [method_failure[m] for m in methods]

        x = range(len(methods))
        width = 0.35

        ax.bar(x, successes, width, label='Successes', color='green')
        ax.bar([i + width for i in x], failures, width, label='Failures', color='red')

        ax.set_title(f"Iteration {iteration}")
        ax.set_ylabel("Count")
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Mission Successes and Failures by Method per Iteration", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("cache/success_failure_plot.png")

def load_all_cached_stats(cache_dir="cache"):
    cached_stats = {}

    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cache directory '{cache_dir}' does not exist.")

    for filename in sorted(os.listdir(cache_dir)):
        match = re.match(r"all_stats_iter_(\d+)\.pkl", filename)
        if match:
            iter_num = int(match.group(1))
            file_path = os.path.join(cache_dir, filename)
            with open(file_path, "rb") as f:
                try:
                    stats = pickle.load(f)
                    cached_stats[iter_num] = stats
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return cached_stats

def plot_cached_stats_subplots(cached_stats_dict):
    num_iterations = len(cached_stats_dict)
    if num_iterations == 0:
        print("No cached data to plot.")
        return

    cols = 2
    rows = (num_iterations + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, (iteration, stat_list) in enumerate(sorted(cached_stats_dict.items())):
        ax = axes[idx]
        method_names = []
        epsilons = []

        for stat in stat_list:
            method = stat.get_method_name()
            eps = stat.get_epsilon_points()
            method_names.extend([method] * len(eps))
            epsilons.extend(eps)

        data = pd.DataFrame({"Method": method_names, "Epsilon": epsilons})
        sns.boxplot(data=data, x="Method", y="Epsilon", ax=ax, palette="Set2")
        ax.set_title(f"Config {iteration}")
        ax.set_ylabel("Epsilon")
        ax.set_xlabel("Method")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 0.9)

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Epsilon Distribution per Oversampling Method by Iteration", fontsize=16)
    plt.savefig("cache/oversampling_stats_plot.png")

if __name__ == "__main__":
    cached_stats = load_all_cached_stats()
    cached_stat = cached_stats[4]
    stats_len = 10
    stats = {}
    for i in range(4):
        stats[i] = cached_stat[i * stats_len: (i+1) * stats_len]
    if cached_stats:
        plot_success_failure_subplots(stats)
        plot_cached_stats_subplots(stats)
    else:
        print("No cached stats found to plot.")