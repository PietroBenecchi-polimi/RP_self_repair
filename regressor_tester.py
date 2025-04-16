from main import run_oversampling_pipeline
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from typing import List, Dict, Any
import numpy as np
import seaborn as sns
from utils.rp_logger import logger
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

def run_experiments(regressor_points: List[int], resampling_points: List[int], invalid_configs_validation: bool) -> List[Dict]:
    """Run the oversampling pipeline for different parameter combinations."""
    save_path = f"tester/data/oversampling_results_{'invalid_configs' if invalid_configs_validation else "first_configs"}.json"
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
                n_data_to_verify=100,
                n_samples=s_points,
                final_validation_invalid_configs=invalid_configs_validation,
                points_regressor=r_points,
                skip_cache=True
            )

            stats_per_points.append({
                'regressor_points': r_points,
                'resampling_points': s_points,
                'stats': stats
            })
            save_results(stats_per_points, save_path)

    return stats_per_points

def process_results(stats_per_points: List[Dict]) -> pd.DataFrame:
    """Process results into a DataFrame suitable for visualization."""
    data = []
    
    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']
        
        for stat in experiment['stats']:
            method = stat['method']
            if 'epsilon_array' in stat:
                for epsilon in stat['epsilon_array']:  # Process each epsilon value individually
                    data.append({
                        'Method': method,
                        'Regressor Points': r_points,
                        'Resampling Points': s_points,
                        'Epsilon': epsilon  # Individual epsilon values
                    })
    
    return pd.DataFrame(data)

def visualize_comparison_violin(df_invalid: pd.DataFrame, df_standard: pd.DataFrame) -> None:
    """Create side-by-side violin plots comparing both validation methods."""
    plt.figure(figsize=(16, 8))
    
    # Create subplots
    plt.subplot(1, 2, 1)
    ax1 = sns.violinplot(data=df_invalid, x='Method', y='Epsilon', 
                        inner='quartile', cut=0, palette='muted', saturation=0.75)
    add_mean_lines(ax1, df_invalid)
    plt.title("Validation with Invalid Configs", fontsize=14, pad=15)
    plt.xlabel("Oversampling Method", fontsize=12)
    plt.ylabel("Epsilon Value", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    ax2 = sns.violinplot(data=df_standard, x='Method', y='Epsilon', 
                        inner='quartile', cut=0, palette='muted', saturation=0.75)
    add_mean_lines(ax2, df_standard)
    plt.title("Standard Validation", fontsize=14, pad=15)
    plt.xlabel("Oversampling Method", fontsize=12)
    plt.ylabel("Epsilon Value", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("oversampling_comparison_violin_plots.png", 
               dpi=300, 
               bbox_inches='tight',
               transparent=False)
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

def main():
    # Configuration parameters
    regressor_points = [5, 10, 30, 50]
    resampling_points = [30, 50, 100]
    
    # Run experiments for both validation types
    standard_stats = run_experiments(regressor_points, resampling_points, False)
    invalid_stats = run_experiments(regressor_points, resampling_points, True)
    
    # Process results
    df_standard = process_results(standard_stats)
    df_invalid = process_results(invalid_stats)
    
    # Add validation type identifier
    df_invalid['Validation Type'] = 'Invalid Configs'
    df_standard['Validation Type'] = 'Standard'
    
    # Visualize comparison
    visualize_comparison_violin(df_invalid, df_standard)
    
    # Optional: Also create a combined plot
    combined_df = pd.concat([df_invalid, df_standard])
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=combined_df, x='Method', y='Epsilon', hue='Validation Type',
                  split=True, inner='quartile', palette='muted')
    plt.title("Epsilon Distribution by Method and Validation Type", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tester/figs/combined_validation_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()