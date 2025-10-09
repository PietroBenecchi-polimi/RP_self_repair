from typing import List, Dict, Any
import pandas as pd
from itertools import zip_longest
import pickle
from self_repair.stats import Stat


def process_results(stats_per_points: List[Dict]) -> pd.DataFrame:
    """Flatten the list of stats into a DataFrame suitable for boxplotting."""
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']
        stats: list[Stat] = experiment['stats']

        for stat in stats:
            method = stat.get_method_name()
            print(f"Processing method: {method}.")
            epsilon_points = stat.get_epsilon_points()

            opt_vals = getattr(stat, "neighbours_optimized", [])
            val_vals = getattr(stat, "neighbours_validation", [])

            if(method != 'target' and method != 'target-20-neighbours'):
                opt_vals = opt_vals['SCS'].to_list()
                val_vals = val_vals['SCS'].to_list()

            if(isinstance(epsilon_points, list)):
                for opt_s, val_s in zip(opt_vals, val_vals):
                    data.append({
                        'Method': method,
                        'Regressor Points': r_points,
                        'Resampling Points': s_points,
                        'Neighbours_optimized': opt_s, 
                        'Neighbours_validation': val_s 
                    })
            elif(isinstance(epsilon_points, Dict)):
                for opt_s, val_s in zip(opt_vals, val_vals):
                    data.append({
                        'Method': method,
                        'Regressor Points': r_points,
                        'Resampling Points': s_points,
                        'Neighbours_optimized': opt_s, 
                        'Neighbours_validation': val_s 
                    })

    return pd.DataFrame(data)

if __name__ == "__main__":
    pkl_path = 'visualization/data/data_paper_pietro_mc/oversampling_results_invalid_configs.pkl'
    with open(pkl_path, "rb") as f:
        stats_per_points = pickle.load(f)

    df = process_results(stats_per_points)
    df.to_csv('processed_oversampling_results_invalid_configs.csv', index=False)
