import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import set_loglevel
import seaborn as sns
from PIL.PngImagePlugin import logger as pil_logger
from typing import Dict
from utils.rp_logger import logger
import logging
import matplotlib.pyplot as plt
from matplotlib import set_loglevel
from scipy.stats import mannwhitneyu
import math

set_loglevel("WARNING")

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True
pil_logger.disabled = True

def plot_single_config_oversampling(invalid_stats, test_name: str) -> None:

    ## process data for visualization
    processed_data = {}

    for i in range(math.floor(len(invalid_stats[0]["stats"])/ 10)):
        processed_data[i] = invalid_stats[0]["stats"][i * 10:(i + 1) * 10 ]

    for stats in processed_data.values():
        for stat in stats:
            if(isinstance(stat.epsilon_points, list)):
                stat.epsilon_points = [point for point in stat.epsilon_points]
            elif(isinstance(stat.epsilon_points, Dict)):
                stat.epsilon_points = [point for point in stat.epsilon_points.values()]


    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)

    ## create grid: single row and boxplot for neighbours
    cols = 2
    rows = (len(stats) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
    axes = axes.flatten()

    # print each stat in a specific boxplot 
    for idx, (iteration, stat_list) in enumerate(sorted(processed_data.items())):
        ax = axes[idx]
        method_names = []
        epsilons = []

        # using target_neighbours as baseline
        baseline_stat = stat_list[1]
        baseline_eps = baseline_stat.get_epsilon_points()

        for indx, stat in enumerate(stat_list):
            method = stat.get_method_name()
            eps = stat.get_epsilon_points()

            # perform Mann–Whitney U test only on neighbours one
            if stat is not baseline_stat and (indx % 2 == 1):
                _, p_value = mannwhitneyu(eps, baseline_eps, alternative='less')

                if p_value < 0.05:
                    method += " P"
                else:
                    method += " F"

            method_names.extend([method] * len(eps))
            epsilons.extend(eps)

        data = pd.DataFrame({"Method": method_names, "Epsilon": epsilons})

        sns.boxplot(data = data, x="Method", y="Epsilon", ax = ax, palette="Set2")

        ax.set_title(f"Config {iteration}")
        ax.set_ylabel("Epsilon")
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylim(0, 0.9)

    # remove unsed subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Epsilon Distribution per Oversampling Method by Iteration", fontsize=16)
    plt.savefig(f"boxplot.png")


def plot_single_config_oversampling(invalid_stats, test_name: str) -> None:

     ## process data for visualization
    processed_data = {}

    for i in range(math.floor(len(invalid_stats[0]["stats"])/ 10)):
        processed_data[i] = invalid_stats[0]["stats"][i * 10:(i + 1) * 10 ]

    for stats in processed_data.values():
        for stat in stats:
            if(isinstance(stat.epsilon_points, list)):
                stat.epsilon_points = [point for point in stat.epsilon_points]
            elif(isinstance(stat.epsilon_points, Dict)):
                stat.epsilon_points = [point for point in stat.epsilon_points.values()]


    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)
    num_iterations = len(stats)
    if num_iterations == 0:
        print("No cached data to plot.")
        return

    cols = 2
    rows = (num_iterations + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, (iteration, stat_list) in enumerate(sorted(processed_data.items())):
        ax = axes[idx]
        method_names = []
        epsilons = []

        baseline_stat = stat_list[1]
        baseline_eps = baseline_stat.get_epsilon_points()
        baseline_name = baseline_stat.get_method_name()

        result_summary = []  # Table content

        for i, stat in enumerate(stat_list):
            method = stat.get_method_name()
            eps = stat.get_epsilon_points()

            # Skip test for baseline and only for neighbours, so odd numbers
            if stat is not baseline_stat and (i % 2 == 1):
                _, p_value = mannwhitneyu(eps, baseline_eps, alternative='less')
                result = "✓" if p_value < 0.05 else "✗"
                result_summary.append(f"{method}: {result} (p-value: {p_value:.4f})")

            method_names.extend([method] * len(eps))
            epsilons.extend(eps)

        data = pd.DataFrame({"Method": method_names, "Epsilon": epsilons})
        sns.boxplot(data=data, x="Method", y="Epsilon", ax=ax, palette="Set2")
        ax.set_title(f"Config {iteration}")
        ax.set_ylabel("Epsilon")
        ax.set_xlabel("Method")
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylim(0, 0.9)

        # Write table with test results on the right side
        table_text = "\n".join(result_summary)
        ax.text(1.05, 0.5, table_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Epsilon Distribution per Oversampling Method by Iteration", fontsize=16)
    plt.savefig(f"boxplot.png")


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu

def plot_allConfigs_boxplot(df_combined: pd.DataFrame, test_name: str) -> None:
    """Crea boxplot per ogni combinazione Regressor+Resampling e una tabella con Mann-Whitney U test."""
    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)
    configs = df_combined[['Regressor Points', 'Resampling Points']].drop_duplicates()
    total = len(configs)

    logger.info(f"Generating {total} box plots...")

    for idx, (_, row) in enumerate(configs.iterrows(), start=1):
        r_points = row['Regressor Points']
        s_points = row['Resampling Points']

        df_sub = df_combined[
            (df_combined['Regressor Points'] == r_points) &
            (df_combined['Resampling Points'] == s_points)
        ]
        
        # get the methods and reference values
        methods_name = list(df_sub['Method'].unique())
        ref = methods_name[0]
        ref_vals = df_sub[df_sub['Method'] == ref]['Epsilons']

        # Mann–Whitney U test
        results = {}
        for m in methods_name:
            if m == ref:
                results[m] = True
            else:
                vals = df_sub[df_sub['Method'] == m]['Epsilons']
                _, p = mannwhitneyu(vals, ref_vals, alternative='less')
                results[m] = p < 0.05
            
        # Tabella come DataFrame per display
        df_res = pd.DataFrame({
            'Method': methods_name,
            'Passed': ['✔️' if results[m] else '✖️' for m in methods_name]
        })

        # Imposta figura e layout
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

        # --- Subplot 1: Boxplot ---
        ax_plot = plt.subplot(gs[0])
        sns.boxplot(
            data=df_sub, x='Method', y='Epsilons',
            hue='Validation Type', palette='muted',
            fliersize=5, linewidth=2, ax=ax_plot
        )

        # median value
        for m in methods_name:
            for vt in df_sub['Validation Type'].unique():
                sub = df_sub[(df_sub['Method'] == m) & (df_sub['Validation Type'] == vt)]
                if sub.empty: continue
                meanv = sub['Epsilons'].mean()
                xpos = methods_name.index(m)
                offset = -0.2 if vt == 'Invalid Configs' else 0.2
                ax_plot.plot([xpos + offset - 0.05, xpos + offset + 0.05],
                             [meanv, meanv], color='red', linewidth=2)

        ax_plot.set_title(f"Validation Comparison — Regressor: {r_points}, Resampling: {s_points}")
        ax_plot.set_xlabel("Oversampling Method")
        ax_plot.set_ylabel("Epsilons")
        ax_plot.set_xticklabels(ax_plot.get_xticklabels(), rotation=45)
        ax_plot.grid(True, axis='y', linestyle=':', alpha=0.7)
        ax_plot.legend(title='Validation Type', loc='upper left')

        # test table
        ax_table = plt.subplot(gs[1])
        ax_table.axis('off')
        table = ax_table.table(cellText=df_res.values,
                               colLabels=df_res.columns,
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.4)

        # save and logger 
        save_path = f"visualization/figs/figs_{test_name}/combined_box_r{r_points}_s{s_points}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"[{idx}/{total}] Saved box plot: {save_path}")


def plot_mean_epsilon_per_method(df_combined: pd.DataFrame, test_name: str):
    """Plot mean epsilon per method for each regressor point."""
    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)

    regressor_values = sorted(df_combined['Regressor Points'].unique())
    total = len(regressor_values)
    logger.info(f"Generating {total} mean epsilon plots...")

    for idx, r_point in enumerate(regressor_values, start=1):
        df_subset = df_combined[df_combined['Regressor Points'] == r_point]

        mean_eps = (
            df_subset
            .groupby(['Method', 'Resampling Points'])['Epsilons']
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=mean_eps,
            x='Resampling Points',
            y='Epsilons',
            hue='Method',
            marker='o'
        )

        plt.title(f"Mean Epsilon per Method — Regressor Points: {r_point}")
        plt.xlabel("Resampling Points")
        plt.ylabel("Mean Epsilon")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(title='Method')
        plt.tight_layout()

        save_path = f"visualization/figs/figs_{test_name}/mean_epsilon_r{r_point}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"[{idx}/{total}] Saved mean epsilon plot: {save_path}")

def plot_variance_epsilon_per_method(df_combined: pd.DataFrame, test_name: str):
    """Plot epsilon variance per method for each regressor point."""
    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)

    regressor_values = sorted(df_combined['Regressor Points'].unique())
    total = len(regressor_values)
    logger.info(f"Generating {total} epsilon variance plots...")

    for idx, r_point in enumerate(regressor_values, start=1):
        df_subset = df_combined[df_combined['Regressor Points'] == r_point]

        grouped = (
            df_subset
            .groupby(['Method', 'Resampling Points'], as_index=False)['Epsilons']
            .var()
            .rename(columns={'Epsilons': 'Epsilon Variance'})
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=grouped,
            x='Resampling Points',
            y='Epsilon Variance',
            hue='Method',
            marker='o',
            linewidth=2
        )

        plt.title(f"Epsilon Variance vs Resampling Points — Regressor Points: {r_point}")
        plt.xlabel("Resampling Points")
        plt.ylabel("Epsilon Variance")
        plt.xticks(sorted(df_subset['Resampling Points'].unique()))
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(title='Method', loc='best')
        plt.tight_layout()

        save_path = f"visualization/figs/figs_{test_name}/variance_epsilon_r{r_point}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"[{idx}/{total}] Saved variance epsilon plot: {save_path}")

