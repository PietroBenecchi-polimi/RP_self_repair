import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple
from self_repair.stats import Stat
from visualization.visualization import process_oversampling_results
from utils.saving_data import load_existing_results

import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_stats_grouped(df: pd.DataFrame, output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)

    # Group by resampling and regressor points
    grouped = df.groupby(['Regressor Points', 'Resampling Points'])

    for (reg_points, res_points), group_df in grouped:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 3]})

        # Left plot: mission success/failure per method
        mission_data = group_df.groupby('Method')[['Mission Success', 'Mission Failed']].sum().reset_index()
        mission_data_melted = mission_data.melt(id_vars='Method', var_name='Mission Status', value_name='Count')

        sns.barplot(data=mission_data_melted, x='Count', y='Method', hue='Mission Status', ax=axes[0])
        axes[0].set_title('Mission Outcome per Method')
        axes[0].legend(loc='lower right')

        # Right plot: Epsilon boxplot per method
        expanded_rows = []
        for _, row in group_df.iterrows():
            for eps in row['Epsilon']:
                expanded_rows.append({
                    'Method': row['Method'],
                    'Epsilon': eps
                })
        eps_df = pd.DataFrame(expanded_rows)
        sns.boxplot(data=eps_df, x='Method', y='Epsilon', palette='muted', fliersize=5, linewidth=2, ax=axes[1])
        axes[1].set_title('Epsilon Distribution per Method')
        axes[1].tick_params(axis='x', rotation=45)

        fig.suptitle(f'Regressor: {reg_points} pts, Resampling: {res_points} pts', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        filename = f"epsilon_mission_plot_reg{reg_points}_res{res_points}.png"
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

if __name__ == "__main__":
    data = load_existing_results("visualization\data\data_test_multi_a10\oversampling_results_first_verification.pkl")
    data = process_oversampling_results(data, "first_verification")
    visualize_stats_grouped(data)