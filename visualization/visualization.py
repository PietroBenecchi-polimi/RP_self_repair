import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from matplotlib import cm
from utils.rp_logger import logger

## What is validation type? And why should i pass it as an argument?
def process_oversampling_results(stats_per_points: List[Dict], validation_type: str) -> pd.DataFrame:
    """Process oversampling experiment results into a DataFrame for visualization."""
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']

        for stat in experiment['stats']:
            method = stat[0]
            epsilon = stat[1]
            data.append({
                'Method': method,
                'Regressor Points': r_points,
                'Resampling Points': s_points,
                'Epsilon': epsilon,
                'Validation Type': validation_type
            })

    return pd.DataFrame(data)

def visualize_comparison_box(df_invalid: pd.DataFrame, df_standard: pd.DataFrame, r_points: int, s_points: int, test_name: str) -> None:
    """Create a box plot showing both validation types for a specific configuration."""
    df_invalid = df_invalid[(df_invalid['Regressor Points'] == r_points) & (df_invalid['Resampling Points'] == s_points)]
    df_standard = df_standard[(df_standard['Regressor Points'] == r_points) & (df_standard['Resampling Points'] == s_points)]

    combined_df = pd.concat([df_invalid, df_standard])

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=combined_df, x='Method', y='Epsilon', hue='Validation Type',
                     palette='muted', fliersize=5, linewidth=2)

    # Optional: Add mean lines for each method/validation combo
    for method in combined_df['Method'].unique():
        for val_type in combined_df['Validation Type'].unique():
            subset = combined_df[(combined_df['Method'] == method) & (combined_df['Validation Type'] == val_type)]
            mean_val = subset['Epsilon'].mean()
            xpos = list(combined_df['Method'].unique()).index(method)
            offset = -0.2 if val_type == 'Invalid Configs' else 0.2
            ax.plot([xpos + offset - 0.05, xpos + offset + 0.05],
                    [mean_val, mean_val],
                    color='red', linewidth=2)

    plt.title(f"Validation Comparison — Regressor: {r_points}, Resampling: {s_points}", fontsize=14)
    plt.xlabel("Oversampling Method", fontsize=12)
    plt.ylabel("Epsilon", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.legend(title='Validation Type')
    plt.tight_layout()
    os.makedirs(f"tester_results/figs/figs_{test_name}", exist_ok=True)
    plt.savefig(f"tester_results/figs/figs_{test_name}/combined_box_r{r_points}_s{s_points}.png", dpi=300)
    plt.show()

def plot_epsilon_vs_regressors_per_sample(df_invalid: pd.DataFrame, df_standard: pd.DataFrame, test_name: str) -> None:
    """Plot epsilon vs. regressor points for each resampling point (s_points), one plot per s_points,
    with fixed colors for specific methods and dynamic colors for others.
    Excludes 'regressor - no oversampling' from plots."""

    combined_df = pd.concat([df_invalid, df_standard])

    # Exclude 'regressor - no oversampling'
    combined_df = combined_df[combined_df["Method"] != "regressor - no oversampling"]

    all_s_points = sorted(combined_df['Resampling Points'].unique())

    # Define style maps for validation types
    style_map = {
        "Standard": {'linestyle': 'solid', 'marker': 'o'},
        "Invalid Configs": {'linestyle': 'dotted', 'marker': 'x'}
    }

    # Fixed color assignments (excluding regressor - no oversampling)
    fixed_colors = {
        "regressor - second verification": "blue",
        "target - no oversampling": "red"
    }

    for s_points in all_s_points:
        df_filtered = combined_df[combined_df["Resampling Points"] == s_points]

        # Aggregate min, max, mean per (method, validation type, regressor points)
        grouped = df_filtered.groupby(['Method', 'Validation Type', 'Regressor Points'])
        summary = grouped['Epsilon'].agg(['min', 'max', 'mean']).reset_index()
        summary['label'] = summary['Method'] + " (" + summary['Validation Type'] + ")"

        # Create unique labels and extract base method names
        unique_labels = summary['label'].unique()
        method_names = summary['Method'].unique()

        # Assign colors
        dynamic_methods = [m for m in method_names if m not in fixed_colors]
        dynamic_cmap = cm.get_cmap("tab10", len(dynamic_methods))
        dynamic_color_map = {m: dynamic_cmap(i) for i, m in enumerate(dynamic_methods)}

        # Combine fixed and dynamic colors
        full_color_map = {**dynamic_color_map, **fixed_colors}

        # Map label -> color using method name
        label_color_map = {
            label: full_color_map[label.split(" (")[0]]
            for label in unique_labels
        }

        plt.figure(figsize=(12, 6))

        for label, group_df in summary.groupby('label'):
            _ , val_type = label.rsplit(' (', 1)
            val_type = val_type.rstrip(')')

            x = group_df['Regressor Points']
            y = group_df['mean']
            y_min = group_df['min']
            y_max = group_df['max']

            style = style_map.get(val_type, {'linestyle': 'solid', 'marker': 'o'})
            color = label_color_map[label]

            plt.plot(x, y, label=label, marker=style['marker'],
                     linestyle=style['linestyle'], color=color)
            plt.fill_between(x, y_min, y_max, alpha=0.15, color=color)

        plt.title(f"Epsilon vs Regressor Points\n(Resampling Points = {s_points})", fontsize=14)
        plt.xlabel("Regressor Points", fontsize=12)
        plt.ylabel("Epsilon", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Method / Validation', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        os.makedirs(f"tester_results/figs/figs_{test_name}", exist_ok=True)
        plt.savefig(f"tester_results/figs/figs_{test_name}/epsilon_vs_regressors_s{s_points}.png", dpi=300)
        plt.close()

def plot_epsilon_vs_samples_per_method(df_invalid: pd.DataFrame, df_standard: pd.DataFrame, test_name: str) -> None:
    """Create subplots for each method showing epsilon vs resampling points,
    with one line per regressor point."""

    combined_df = pd.concat([df_invalid, df_standard])
    
    combined_df = combined_df[combined_df["Method"] != "regressor - no oversampling"]
    combined_df = combined_df[combined_df["Method"] != "target - no oversampling"]
    combined_df = combined_df[combined_df["Method"] != "regressor - second verification"]

    methods = sorted(combined_df['Method'].unique())
    validation_types = combined_df['Validation Type'].unique()
    n_methods = len(methods)

    # Create a color palette for regressor points
    regressor_points = sorted(combined_df["Regressor Points"].unique())
    palette = sns.color_palette("viridis", len(regressor_points))
    color_map = {r: palette[i] for i, r in enumerate(regressor_points)}

    n_cols = 2
    n_rows = int(np.ceil(n_methods / n_cols))

    palette = sns.color_palette("Set2", len(regressor_points))
    color_map = {r: palette[i % len(palette)] for i, r in enumerate(regressor_points)}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]
        for val_type in validation_types:
            df_sub = combined_df[
                (combined_df['Method'] == method) &
                (combined_df['Validation Type'] == val_type)
            ]

            grouped = df_sub.groupby(['Resampling Points', 'Regressor Points'])['Epsilon'].agg(['min', 'max', 'mean']).reset_index()

            for r_point in regressor_points:
                r_data = grouped[grouped['Regressor Points'] == r_point]
                if r_data.empty:
                    continue
                x = r_data['Resampling Points']
                y = r_data['mean']
                y_min = r_data['min']
                y_max = r_data['max']

                linestyle = '-' if val_type == "Standard" else '--'
                label = f"R={r_point} ({val_type})"

                ax.plot(x, y, label=label, linestyle=linestyle, color=color_map[r_point])
                ax.fill_between(x, y_min, y_max, color=color_map[r_point], alpha=0.15)

        ax.set_title(method, fontsize=12)
        ax.set_xlabel("Resampling Points", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Epsilon", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8, loc='best')

    # Remove unused subplots if any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Epsilon vs Resampling Points\nPer Method and Validation Type", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(f"tester_results/figs/figs_{test_name}", exist_ok=True)
    plt.savefig(f"tester_results/figs/figs_{test_name}/epsilon_vs_samples_per_method.png", dpi=300)
    plt.show()


def add_mean_lines(ax, df):
    """Helper function to add mean lines and annotations to a plot."""
    means = df.groupby('Method')['Epsilon'].mean()
    for i, method in enumerate(means.index):
        ax.hlines(means[method], 
                 xmin=i-0.4, xmax=i+0.4, 
                 colors='red', 
                 linestyles='dashed', 
                 linewidth=2.5, 
                 label='Mean' if i == 0 else "",
                 zorder=3)
    
    for i, (method, mean_val) in enumerate(means.items()):
        ax.text(i, mean_val + 0.02, f'{mean_val:.2f}',
               horizontalalignment='center',
               color='red',
               weight='semibold',
               fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', framealpha=1)

def visualize_comparison_violin(df_invalid: pd.DataFrame, df_standard: pd.DataFrame, r_points: int, s_points: int, test_name:str) -> None:
    """Create a split violin plot showing both validation types for a specific configuration."""
    df_invalid = df_invalid.copy()
    df_standard = df_standard.copy()

    combined_df = pd.concat([df_invalid, df_standard])

    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(data=combined_df, x='Method', y='Epsilon', hue='Validation Type',
                        split=True, inner='quartile', palette='muted', cut=0, saturation=0.75)

    # Add mean lines per method and validation type
    for method in combined_df['Method'].unique():
        for val_type in ['Invalid Configs', 'Standard']:
            mean_val = combined_df[(combined_df['Method'] == method) & 
                                   (combined_df['Validation Type'] == val_type)]['Epsilon'].mean()
            xpos = list(combined_df['Method'].unique()).index(method)
            offset = -0.2 if val_type == 'Invalid Configs' else 0.2
            ax.plot([xpos + offset - 0.05, xpos + offset + 0.05],
                    [mean_val, mean_val],
                    color='red', linewidth=2)
    
    plt.title(f"Validation Comparison — Regressor: {r_points}, Resampling: {s_points}", fontsize=14)
    plt.xlabel("Oversampling Method", fontsize=12)
    plt.ylabel("Epsilon", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.legend(title='Validation Type')
    plt.tight_layout()
    os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)
    plt.savefig(f"visualization/figs/figs_{test_name}/combined_violin_r{r_points}_s{s_points}.png", dpi=300)
    plt.show()
    
def visualize_comparison_box(df_invalid: pd.DataFrame, df_standard: pd.DataFrame, r_points: int, s_points: int, test_name: str) -> None:
    """Create a box plot showing both validation types for a specific configuration."""
    df_invalid = df_invalid.copy()
    df_standard = df_standard.copy()

    # Combine the dataframes for the two validation types
    combined_df = pd.concat([df_invalid, df_standard])

    plt.figure(figsize=(12, 6))

    # Create a box plot with the 'Method' on the x-axis, 'Epsilon' on the y-axis, 
    # and 'Validation Type' as the hue for color differentiation.
    ax = sns.boxplot(data=combined_df, x='Method', y='Epsilon', hue='Validation Type',
                     palette='muted', fliersize=5, linewidth=2)

    # Add mean lines per method and validation type
    for method in combined_df['Method'].unique():
        for val_type in ['Invalid Configs', 'Standard']:
            mean_val = combined_df[(combined_df['Method'] == method) & 
                                   (combined_df['Validation Type'] == val_type)]['Epsilon'].mean()
            xpos = list(combined_df['Method'].unique()).index(method)
            offset = -0.2 if val_type == 'Invalid Configs' else 0.2
            ax.plot([xpos + offset - 0.05, xpos + offset + 0.05],
                    [mean_val, mean_val],
                    color='red', linewidth=2)

    plt.title(f"Validation Comparison — Regressor: {r_points}, Resampling: {s_points}", fontsize=14)
    plt.xlabel("Oversampling Method", fontsize=12)
    plt.ylabel("Epsilon", fontsize=12)
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
                y='Epsilon',
                hue='Method',
                markers=True,
                dashes=False,
                ax=ax
            )

            ax.set_title(f"{v_type} - Regressor Points: {r_point}", fontsize=13)
            ax.set_xlabel("Resampling Points")
            ax.set_ylabel("Epsilon")
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Epsilon vs. Resampling Points ({v_type})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"epsilon_trend_{v_type.replace(' ', '_').lower()}"
        os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)
        plt.savefig(f"visualization/figs/figs_{test_name}/{filename}.png", dpi=300)