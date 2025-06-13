import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from matplotlib import cm
<<<<<<< HEAD
from utils.rp_logger import logger
=======
>>>>>>> test_mono
from self_repair.stats import Stat

def process_results(stats_per_points: List[Dict]) -> pd.DataFrame:
    """Flatten the list of stats into a DataFrame suitable for boxplotting."""
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']
        stats: list[Stat] = experiment['stats']
<<<<<<< HEAD
        for stat in stats:
            method = stat.get_method_name()
            data.append({
                'Method': method,
                'Regressor Points': r_points,
                'Resampling Points': s_points,
                'Epsilon': stat.get_epsilon_points(),
                'Mission Success': stat.get_n_mission_success(),
                'Mission Failed': stat.get_n_mission_failed(),
                'Validation Type': validation_type
            })
=======

        for stat in stats:
            method = stat.get_method_name()
            for eps in stat.get_epsilon_points():
                data.append({
                    'Method': method,
                    'Regressor Points': r_points,
                    'Resampling Points': s_points,
                    'Epsilons': eps
                })
>>>>>>> test_mono

    return pd.DataFrame(data)


def visualize_comparison_box(df_invalid: pd.DataFrame, r_points: int, s_points: int, test_name: str, df_standard: pd.DataFrame = pd.DataFrame([])) -> None:
    """Create a box plot showing both validation types for a specific configuration."""
    df_invalid = df_invalid.copy()
    df_standard = df_standard.copy()

    # Combine the dataframes for the two validation types
    combined_df = pd.concat([df_invalid, df_standard])

    plt.figure(figsize=(12, 6))

    # Create a box plot with the 'Method' on the x-axis, 'Epsilons' on the y-axis, 
    # and 'Validation Type' as the hue for color differentiation.
    ax = sns.boxplot(data=combined_df, x='Method', y='Epsilons', hue='Validation Type',
                     palette='muted', fliersize=5, linewidth=2)

    # Add mean lines per method and validation type
    for method in combined_df['Method'].unique():
        for val_type in ['Invalid Configs', 'Standard']:
            mean_val = combined_df[(combined_df['Method'] == method) & 
                                   (combined_df['Validation Type'] == val_type)]['Epsilons'].mean()
            xpos = list(combined_df['Method'].unique()).index(method)
            offset = -0.2 if val_type == 'Invalid Configs' else 0.2
            ax.plot([xpos + offset - 0.05, xpos + offset + 0.05],
                    [mean_val, mean_val],
                    color='red', linewidth=2)

    plt.title(f"Validation Comparison — Regressor: {r_points}, Resampling: {s_points}", fontsize=14)
    plt.xlabel("Oversampling Method", fontsize=12)
    plt.ylabel("Epsilons", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.legend(title='Validation Type')
    plt.tight_layout()
    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)
    plt.savefig(f"visualization/figs/figs_{test_name}/combined_box_r{r_points}_s{s_points}.png", dpi=300)
    plt.show()

def plot_epsilon_over_resampling_points(df_combined: pd.DataFrame, test_name: str):
    """Plot epsilon vs. resampling points separately for each validation type."""
    validation_types = df_combined['Validation Type'].unique()

    for v_type in validation_types:
        df_subset = df_combined[df_combined['Validation Type'] == v_type]
        regressor_points = sorted(df_subset['Regressor Points'].unique())

        n_rows = int(np.ceil(len(regressor_points) / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows), sharey=True)
        axes = axes.flatten()

        for idx, r_point in enumerate(regressor_points):
            ax = axes[idx]
            subset = df_subset[df_subset['Regressor Points'] == r_point]

            sns.lineplot(
                data=subset,
                x='Resampling Points',
                y='Epsilons',
                hue='Method',
                markers=True,
                dashes=False,
                ax=ax
            )

            ax.set_title(f"{v_type} - Regressor Points: {r_point}", fontsize=13)
            ax.set_xlabel("Resampling Points")
            ax.set_ylabel("Epsilons")
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Epsilons vs. Resampling Points ({v_type})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"epsilon_trend_{v_type.replace(' ', '_').lower()}"
        os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)
        plt.savefig(f"visualization/figs/figs_{test_name}/{filename}.png", dpi=300)