from self_repair.self_repair import run_oversampling_pipeline
import os
import sys
import pickle
from typing import List, Dict
from utils.rp_logger import logger
import utils.datacleaner as dc


def run_experiments(regressor_points: List[int], resampling_points: List[int], second_test_data_str: str, test_name: str) -> List[Dict]:
    save_path = f"output/data_{test_name}/oversampling_results_{second_test_data_str}.pkl"
    os.makedirs(f"output/data_{test_name}", exist_ok=True)

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
                n_data_to_verify=100,
                n_samples=s_points,
                data_type_second_validation=second_test_data_str,
                points_regressor=r_points,
                max_iterations=30
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
    regressor_points = [100]
    resampling_points = [10] 

    if len(sys.argv) < 2:
       logger.error("Please, insert the test name as an argument. It is used to save the results.")
       return
    
    test_name = sys.argv[1]
    if(len(test_name.split(" ")) > 1):
       logger.error("Invalid test name, please use the suggested format")
       return
    
    # Perform oversampling pipeline:
    invalid_stats = run_experiments(regressor_points, resampling_points, "invalid_configs", test_name=test_name)
    df_invalid = dc.process_results(invalid_stats)
    df_invalid.to_csv(f"output/data_{test_name}/boxplot_data_invalid_configs.csv", index=False)
        
if __name__ == "__main__":
    mono_test_pipeline()
    