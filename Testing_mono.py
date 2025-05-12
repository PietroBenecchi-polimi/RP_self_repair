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
                n_data_to_verify=10,
                n_samples=s_points,
                data_type_second_validation=invalid_configs_validation,
                points_regressor=r_points,
                skip_cache=False,
            )

            stats_per_points.append({
                'regressor_points': r_points,
                'resampling_points': s_points,
                'stats': stats
            })
            save_results(stats_per_points, save_path)

    return stats_per_points

def main():
    regressor_points = [100, 150, 300, 500]
    resampling_points = [10, 15, 30, 50, 100]
    #if len(sys.argv) < 2:
    #    logger.error("Please, insert the correct number of arguments")
    #    return 
    test_name : str = "model_checker"
    #if(len(test_name.split(" ")) > 1):
    #    logger.error("Invalid test name, please use the suggested format")
    #    return
    # Perform oversmapling pipeline:
    # 1. First validation
    # 2. Oversmapling
    # 3. Second validation: It can be invalid configs or standard(first validation)
    standard_stats = run_experiments(regressor_points, resampling_points, "first_verification", test_name=test_name)
    invalid_stats = run_experiments(regressor_points, resampling_points, "invalid_configs", test_name=test_name)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()