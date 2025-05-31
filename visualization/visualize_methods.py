from self_repair.self_repair_multi import run_oversampling_pipeline
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns

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