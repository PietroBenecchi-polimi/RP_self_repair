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

def run_experiments(regressor_points: List[int], resampling_points: List[int], invalid_configs_validation: str) -> List[Dict]:
    """Run the oversampling pipeline for different parameter combinations."""
    
    save_path = f"tester_results/data/oversampling_results_{'invalid_configs' if invalid_configs_validation else 'first_configs'}.json"
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
                data_type_second_validation=invalid_configs_validation,
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

def visualize_comparison_violin(df_invalid: pd.DataFrame, df_standard: pd.DataFrame, r_points: int, s_points: int) -> None:
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
    plt.savefig(f"tester_results/figs/combined_violin_r{r_points}_s{s_points}.png", dpi=300)
    plt.show()
    
def visualize_comparison_box(df_invalid: pd.DataFrame, df_standard: pd.DataFrame, r_points: int, s_points: int) -> None:
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
    plt.savefig(f"tester_results/figs/combined_box_r{r_points}_s{s_points}.png", dpi=300)
    plt.show()

def plot_epsilon_over_resampling_points(df_combined: pd.DataFrame):
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
        filename = f"tester_results/figs/epsilon_trend_{v_type.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300)


def create_maxplot(df_standard: pd.DataFrame):
    """Create maxplot for each oversampling method comparing standard and invalid configs."""
    fig, axes = plt.subplots(df_standard['method'].__len__ , 1, figsize=(10, 8))  # 2x2 grid of subplots

    # Loop through each method and generate a maxplot
    for i, method in enumerate(df_standard['method']):
        epsilon_array = df_standard['stats']['epsilon_array'] 

        # Plot each boxplot in a separate subplot
        axes[i, 0].boxplot(epsilon_array)

        # Add labels and title
        axes[i, 0].title('Boxplot for ' + method)
        axes[i, 0].xlabel('Oversampling Method ' + method)
        axes[i, 0].ylabel('Epsilon')
        
        # Save and show
        filename = f"tester_results/figs/boxplotForResamplingMethods.png"
        plt.savefig(filename, dpi=300)
        plt.show()

def create_maxplot(df_standard: pd.DataFrame):
    """Create maxplot for each oversampling method comparing standard and invalid configs."""
    # Create a figure with a number of subplots equal to the number of methods
    fig, axes = plt.subplots(len(df_standard['method']), 1, figsize=(10, 8))

    # Loop through each method and generate a maxplot
    for i, method in enumerate(df_standard['method']):
        epsilon_array = df_standard['stats'][i]['epsilon_array']  

        # Plot each boxplot in a separate subplot
        axes[i].boxplot(epsilon_array)

        # Add labels and title
        axes[i].set_title(f'Boxplot for {method}')
        axes[i].set_xlabel(f'Oversampling Method: {method}')
        axes[i].set_ylabel('Epsilon')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure with all subplots to a single file
    filename = "tester_results/figs/boxplotForResamplingMethods.png"
    plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()

def main():
    regressor_points = [100, 150, 300, 500]
    resampling_points = [30, 50, 100]

    # Perform oversmapling pipeline:
    # 1. First validation
    # 2. Oversmapling
    # 3. Second validation: It can be invalid configs or standard(first validation)
    standard_stats = run_experiments(regressor_points, resampling_points, "first_verification")
    invalid_stats = run_experiments(regressor_points, resampling_points, "invalid_configs")

    # Create a MaxPlot of epsilon over resampling points

    # Process results into DataFrames
    df_standard = process_results(standard_stats)
    df_invalid = process_results(invalid_stats)
    
    df_invalid['Validation Type'] = 'Invalid Configs'
    df_standard['Validation Type'] = 'Standard'

    for r in regressor_points:
        for s in resampling_points:
            # Filter the DataFrames for the current combination of regressor and resampling points
            df_s = df_standard[(df_standard['Regressor Points'] == r) & (df_standard['Resampling Points'] == s)]
            df_i = df_invalid[(df_invalid['Regressor Points'] == r) & (df_invalid['Resampling Points'] == s)]
            # Check if both DataFrames are not empty before plotting
            if not df_s.empty and not df_i.empty:
                visualize_comparison_violin(df_i, df_s, r_points=r, s_points=s)
                
    # Combine for general plotting
    df_combined = pd.concat([df_standard, df_invalid])
    plot_epsilon_over_resampling_points(df_combined)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
