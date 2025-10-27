"""
Experiment Runner Script
------------------------
This script runs oversampling experiments for regression models using a model checker
or regressor as the ground truth. It generates experiment results and an overview log
containing configuration and environment details.

Usage:
    python script_name.py <test_name>

Output:
    - output/<test_name>/experiment_results.csv : CSV file with experiment results.
    - output/<test_name>/overview.txt : JSON file with configuration and runtime details.
"""

from self_repair.self_repair import run_oversampling_pipeline
import os
import sys
from self_repair.mc_opt_interface import ModelCheckerInterface, RegressorInterface, MC_OPT_INTERFACE
from typing import List, Dict
from utils.rp_logger import logger
import utils.datacleaner as dc
import datetime as dt
import json

def save_experiment_log(output_dir, test_name, regressor_points, resampling_points, ground_truth):
    """
    Saves experiment metadata and configuration details to a JSON-formatted overview file.

    Args:
        output_dir (str): Directory where the log file will be stored.
        test_name (str): Name of the experiment/test.
        regressor_points (List[int]): List of regressor data point configurations.
        resampling_points (List[int]): List of resampling point configurations.
        ground_truth (MC_OPT_INTERFACE): Ground truth model interface instance.
    """
    log_data = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_name": test_name,
        "regressor_points": regressor_points,
        "resampling_points": resampling_points,
        "ground_truth_type": ground_truth.__class__.__name__,
        "cwd": os.getcwd(),
        "python_version": sys.version,
        "script": os.path.basename(__file__)
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "overview.txt"), "w") as f:
        json.dump(log_data, f, indent=4)

def run_experiments(regressor_points: List[int], resampling_points: List[int], ground_truth: MC_OPT_INTERFACE) -> List[Dict]:
    """
    Runs oversampling experiments for all combinations of regressor and resampling points.

    Args:
        regressor_points (List[int]): List of regressor sample sizes to test.
        resampling_points (List[int]): List of resampling sample sizes to test.
        ground_truth (MC_OPT_INTERFACE): Model checker or regressor used as ground truth.

    Returns:
        List[Dict]: List of experiment results with configuration and performance statistics.
    """
    stats_per_points = []
    existing_combinations = {(d['regressor_points'], d['resampling_points']) 
                            for d in stats_per_points if 'regressor_points' in d}

    for r_points in regressor_points:
        for s_points in resampling_points:
            logger.info(f"Running experiment with regressor_points={r_points}, resampling_points={s_points}")
            if (r_points, s_points) in existing_combinations:
                logger.warning(f"Skipping already processed combination: regressor={r_points}, resampling={s_points}")
                continue
            
            stats = run_oversampling_pipeline(
                n_data_to_verify=1,
                n_samples=s_points,
                points_regressor=r_points,
                mc_opt_interface=ground_truth,
                max_iterations=30,
            )

            stats_per_points.append(
                {
                    "regressor_points": r_points,
                    "resampling_points": s_points,
                    "stats": stats
                }
            )

    return stats_per_points

def mono_test_pipeline():
    """
    Runs a single test pipeline with predefined parameters.
    It performs the oversampling experiments, processes the results, 
    and saves both CSV output and a configuration overview file.
    """
    regressor_points = [100]
    resampling_points = [2]
    ground_truth = ModelCheckerInterface()  # Change to RegressorInterface() if needed

    # Check for valid test name argument
    if len(sys.argv) < 2:
        logger.error("Please, insert the test name as an argument. It is used to save the results.")
        return
    
    test_name = sys.argv[1]
    if len(test_name.split(" ")) > 1:
        logger.error("Invalid test name, please use the suggested format")
        return
    
    # Create output directory and run experiment
    os.makedirs(f"output/{test_name}", exist_ok=True)
    invalid_stats = run_experiments(regressor_points, resampling_points, ground_truth)
    df_invalid = dc.process_results(invalid_stats)
    df_invalid.to_csv(f"output/{test_name}/experiment_results.csv", index=False)

    # Save log with configuration details
    save_experiment_log(f"output/{test_name}/", test_name, regressor_points, resampling_points, ground_truth)

if __name__ == "__main__":
    mono_test_pipeline()