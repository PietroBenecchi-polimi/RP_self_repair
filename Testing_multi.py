from self_repair.self_repair_multi import run_oversampling_pipeline
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import sys
from typing import List, Dict
from utils.rp_logger import logger
from visualization.visualize_methods import *

def save_results(stats_per_points: List[Dict], save_path: str) -> None:
    """Save results to a Pickle file."""
    with open(save_path, "wb") as f:
        pickle.dump(stats_per_points, f)

def load_existing_results(save_path: str) -> List[Dict]:
    """Load existing results from a Pickle file."""
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)
    return []

def run_experiments(regressor_points: List[int], resampling_points: List[int], invalid_configs_validation: str, test_name: str) -> List[Dict]:
    """Run the oversampling pipeline for different parameter combinations."""
    save_path = f"visualization/data/data_{test_name}/oversampling_results_{invalid_configs_validation}.pkl"
    os.makedirs(f"visualization/data/data_{test_name}", exist_ok=True)
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
                n_data_to_verify=30,
                n_samples=s_points,
                data_type_second_validation=invalid_configs_validation,
                points_regressor=r_points,
                skip_cache=False
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

def main():
    regressor_points = [100, 150, 300, 500]
    resampling_points = [30, 50, 100]
    test_name : str = sys.argv[1]
    if(len(test_name.split(" ")) > 1):
        logger.error("Invalid test name, please use the suggested format")
        return
    # Perform oversmapling pipeline:
    # 1. First validation
    # 2. Oversmapling
    # 3. Second validation: It can be invalid configs or standard(first validation)
    standard_stats = run_experiments(regressor_points, resampling_points, "first_verification", test_name=test_name)
    invalid_stats = run_experiments(regressor_points, resampling_points, "invalid_configs", test_name=test_name)

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
                visualize_comparison_violin(df_i, df_s, r_points=r, s_points=s, test_name=test_name)
                
    # Combine for general plotting
    df_combined = pd.concat([df_standard, df_invalid])
    plot_epsilon_over_resampling_points(df_combined, test_name=test_name)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
