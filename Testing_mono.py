from self_repair.self_repair_mono import run_oversampling_pipeline
import matplotlib.pyplot as plt
import json
import os
import sys
import pandas as pd
from typing import List, Dict, Any
import numpy as np
import seaborn as sns
from utils.rp_logger import logger
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def save_results(stats_per_points: List[Dict], save_path: str) -> None:
    """Save results to JSON file."""
    with open(save_path, "w") as f:
        json.dump(stats_per_points, f, indent=2)

def load_existing_results(save_path: str) -> List[Dict]:
    """Load existing results from JSON file."""
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)
    return []

def run_experiments(regressor_points: List[int], resampling_points: List[int], invalid_configs_validation: str, test_name: str) -> List[Dict]:
    """Run the oversampling pipeline for different parameter combinations."""
    save_path = f"tester_results/data/data_{test_name}/oversampling_results_{invalid_configs_validation}.json"
    os.makedirs(f"tester_results/data/data_{test_name}", exist_ok=True)
    stats_per_points = load_existing_results(save_path)
    existing_combinations = {(d['regressor_points'], d['resampling_points']) 
                            for d in stats_per_points if 'regressor_points' in d}

    for r_points in regressor_points:
        for s_points in resampling_points:
            if (r_points, s_points) in existing_combinations:
                logger.warning(f"Skipping already processed combination: regressor={r_points}, resampling={s_points}")
                continue

            logger.info(f"\nRunning experiment with regressor_points={r_points}, resampling_points={s_points}")
            stats = run_oversampling_pipeline(
                n_data_to_verify=50,
                n_samples=s_points,
                data_type_second_validation=invalid_configs_validation,
                points_regressor=r_points,
                skip_cache=True,
            )

            stats_per_points.append({
                'regressor_points': r_points,
                'resampling_points': s_points,
                'stats': stats
            })
            save_results(stats_per_points, save_path)

    return stats_per_points

def process_results(stats_per_points: List[Dict], validation_type: str) -> pd.DataFrame:
    """Process results into a DataFrame suitable for visualization."""
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']

        for stat in experiment['stats']:
            method = stat['method']
            epsilon = stat['epsilon']  # Single float value now
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

    # ❌ Exclude 'regressor - no oversampling'
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
            method, val_type = label.rsplit(' (', 1)
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

def classical_oversampling_pipeline():
    regressor_points = [100, 150, 500]
    resampling_points = [10]
    if len(sys.argv) < 2:
       logger.error("Please, insert the correct number of arguments")
       return
    
    test_name = sys.argv[1]
    if(len(test_name.split(" ")) > 1):
       logger.error("Invalid test name, please use the suggested format")
       return
    # Perform oversmapling pipeline:
    # 1. First validation
    # 2. Oversmapling
    # 3. Second validation: It can be invalid configs or standard(first validation)
    standard_stats = run_experiments(regressor_points, resampling_points, "first_verification", test_name=test_name)
    invalid_stats = run_experiments(regressor_points, resampling_points, "invalid_configs", test_name=test_name)
    df_standard = process_results(standard_stats, "Standard")
    df_invalid = process_results(invalid_stats, "Invalid Configs")

    plot_epsilon_vs_regressors_per_sample(df_invalid, df_standard, test_name=test_name)
    plot_epsilon_vs_samples_per_method(df_invalid, df_standard, test_name=test_name)

def oversampling_unsuccessful_mission_pipeline():
    regressor_points = [100, 200 ,300, 500]
    resampling_points = [10, 50, 100, 300]
    if len(sys.argv) < 2:
       logger.error("Please, insert the correct number of arguments")
       return
    
    test_name = sys.argv[1]
    if(len(test_name.split(" ")) > 1):
       logger.error("Invalid test name, please use the suggested format")
       return
    # Perform oversmapling pipeline:
    # 1. First validation
    # 2. Oversmapling
    # 3. Second validation: It can be invalid configs or standard(first validation)
    standard_stats = run_experiments(regressor_points, resampling_points, "first_verification", test_name=test_name)
    invalid_stats = run_experiments(regressor_points, resampling_points, "invalid_configs", test_name=test_name)
    df_standard = process_results(standard_stats, "Standard")
    df_invalid = process_results(invalid_stats, "Invalid Configs")

    plot_epsilon_vs_regressors_per_sample(df_invalid, df_standard, test_name=test_name)
    plot_epsilon_vs_samples_per_method(df_invalid, df_standard, test_name=test_name)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    # Normal oversampling pipeline
    # classical_oversampling_pipeline()
    # Test 1: CHANGE OF AREA DISTRIBUION
    #TODO LUIGI
    # Test 2: oversampling for unsuccessful missions and combination of those
    oversampling_unsuccessful_mission_pipeline()