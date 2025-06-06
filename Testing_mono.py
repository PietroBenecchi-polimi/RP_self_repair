from self_repair.mono_test_1 import run_oversampling_pipeline
import os
import sys
from typing import List, Dict
from utils.rp_logger import logger
import utils.saving_data as sd


def run_experiments(regressor_points: List[int], resampling_points: List[int], second_test_data_str: str, test_name: str) -> List[Dict]:
    save_path = f"tester_results/data/data_{test_name}/oversampling_results_{second_test_data_str}.pkl"
    os.makedirs(f"tester_results/data/data_{test_name}", exist_ok=True)
    stats_per_points_resampling = []

    for r_points in regressor_points:
        for s_points in resampling_points:
            logger.info(f"\nRunning experiment with regressor_points={r_points}, resampling_points={s_points}")

            stats = run_oversampling_pipeline(
                n_data_to_verify=10,
                n_samples=s_points,
                data_type_second_validation=second_test_data_str,
                points_regressor=r_points,
                skip_cache=True,
            )

            stats_per_points_resampling.append(
                {
                    "regressor_points": r_points,
                    "resampling_points": s_points,
                    "stats": stats
                }
            )

            sd.save_results(stats_per_points_resampling, save_path)


def mono_test_pipeline():
    regressor_points = [100, 150, 500]
    resampling_points = [50, 100, 300]

    if len(sys.argv) < 2:
       logger.error("Please, insert the test name as an argument. It is used to save the results.")
       return
    
    test_name = sys.argv[1]
    if(len(test_name.split(" ")) > 1):
       logger.error("Invalid test name, please use the suggested format")
       return
    
    # Perform oversmapling pipeline:
    standard_stats = run_experiments(regressor_points, resampling_points, "first_verification", test_name=test_name)
    invalid_stats = run_experiments(regressor_points, resampling_points, "invalid_configs", test_name=test_name)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    mono_test_pipeline()