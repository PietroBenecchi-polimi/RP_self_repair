from utils.saving_data import load_existing_results
from visualization.visualization import process_results

def load_and_print_results(save_path: str):
    results = load_existing_results(save_path)
    results = process_results(results)

    return results

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots_grouped(df: pd.DataFrame, test_name: str = "default_test"):
    grouped = df.groupby(['Regressor Points', 'Resampling Points'])

    for (r_points, s_points), group in grouped:
        if group.empty:
            continue

        # Unpack epsilons into long-form DataFrame
        records = []
        for _, row in group.iterrows():
            method = row['Method']
            epsilons = row['Epsilons']
            for val in epsilons:
                records.append({
                    'Method': method,
                    'Epsilon': val
                })

        combined_df = pd.DataFrame(records)
        if combined_df.empty:
            continue

        plt.figure(figsize=(12, 6))

        # Seaborn boxplot per method
        ax = sns.boxplot(data=combined_df, x='Method', y='Epsilon',
                         palette='muted', fliersize=5, linewidth=2)

        # Add red mean lines per method
        for method in combined_df['Method'].unique():
            mean_val = combined_df[combined_df['Method'] == method]['Epsilon'].mean()
            xpos = list(combined_df['Method'].unique()).index(method)
            ax.plot([xpos - 0.2, xpos + 0.2], [mean_val, mean_val],
                    color='red', linewidth=2)

        plt.title(f"Epsilon Boxplot — Regressor: {r_points}, Resampling: {s_points}", fontsize=14)
        plt.xlabel("Oversampling Method", fontsize=12)
        plt.ylabel("Epsilon", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()

        # Save the plot
        os.makedirs(f"visualization/figs/figs_{test_name}", exist_ok=True)
        plt.savefig(f"visualization/figs/figs_{test_name}/box_r{r_points}_s{s_points}.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    save_path = "tester_results/data/data_test_mono/oversampling_results_first_verification.pkl"
    results = load_and_print_results(save_path)
    plot_boxplots_grouped(results, test_name="test_mono")