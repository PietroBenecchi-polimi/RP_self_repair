import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import set_loglevel
import seaborn as sns
from PIL.PngImagePlugin import logger as pil_logger
from typing import List, Dict
from self_repair.stats import Stat
from utils.rp_logger import logger
import logging
import matplotlib.pyplot as plt
from matplotlib import set_loglevel
set_loglevel("WARNING")

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True
pil_logger.disabled = True

def process_results(stats_per_points: List[Dict]) -> pd.DataFrame:
    """Flatten the list of stats into a DataFrame suitable for boxplotting."""
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']
        stats: list[Stat] = experiment['stats']

        for stat in stats:
            method = stat.get_method_name()
            for eps in stat.get_epsilon_points():
                data.append({
                    'Method': method,
                    'Regressor Points': r_points,
                    'Resampling Points': s_points,
                    'Epsilons': eps
                })

    return pd.DataFrame(data)


def visualize_comparison_box(df_combined: pd.DataFrame, test_name: str) -> None:
    """Create a box plot for each unique (regressor, resampling) configuration."""
    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)

    configs = df_combined[['Regressor Points', 'Resampling Points']].drop_duplicates()
    total = len(configs)
    logger.info(f"Generating {total} box plots...")

    for idx, (_, row) in enumerate(configs.iterrows(), start=1):
        r_points = row['Regressor Points']
        s_points = row['Resampling Points']

        df_subset = df_combined[
            (df_combined['Regressor Points'] == r_points) &
            (df_combined['Resampling Points'] == s_points)
        ]

        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(
            data=df_subset,
            x='Method',
            y='Epsilons',
            hue='Validation Type',
            palette='muted',
            fliersize=5,
            linewidth=2
        )

        # Add mean lines
        for method in df_subset['Method'].unique():
            for val_type in df_subset['Validation Type'].unique():
                mean_val = df_subset[
                    (df_subset['Method'] == method) & 
                    (df_subset['Validation Type'] == val_type)
                ]['Epsilons'].mean()
                xpos = list(df_subset['Method'].unique()).index(method)
                offset = -0.2 if val_type == 'Invalid Configs' else 0.2
                ax.plot([xpos + offset - 0.05, xpos + offset + 0.05],
                        [mean_val, mean_val], color='red', linewidth=2)

        plt.title(f"Validation Comparison — Regressor: {r_points}, Resampling: {s_points}")
        plt.xlabel("Oversampling Method")
        plt.ylabel("Epsilons")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.legend(title='Validation Type')
        plt.tight_layout()

        save_path = f"visualization/figs/figs_{test_name}/combined_box_r{r_points}_s{s_points}.png"
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