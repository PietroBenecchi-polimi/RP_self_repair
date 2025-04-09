import pandas as pd
import joblib
from oversampling import lime_based_resampling, random_oversampling, smote_oversampling
from validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from typing import Dict
import multiprocessing
import time
import sys
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger
from utils.datacleaner import categorical_to_numeric
from mc_opt_interface import *

def oversample_method(method_name: str, invalid_configs: pd.DataFrame, regressor=None) -> Dict:
    if method_name == "Random":
        new_samples = random_oversampling(df=invalid_configs)
    elif method_name == "Smote":
        new_samples = smote_oversampling(df=invalid_configs)
    elif method_name == "LIME_gaussian":
        new_samples = lime_based_resampling(df=invalid_configs, regressor=regressor)
        new_samples['SCS'] = 0
    else:
        raise ValueError(f"Unknown oversampling method: {method_name}")

    return {
        "method": method_name,
        "samples": new_samples
    }

def oversampling_methods_parallel(invalid_configs: pd.DataFrame, n_samples=100, regressor=None) -> Dict:
    # Samples from invalid configurations
    samples = invalid_configs.sample(n=n_samples) if len(invalid_configs) > n_samples else invalid_configs
    # Create list of oversampling methods
    methods = ["Smote", "Random", "LIME_gaussian"]
    
    # Use multiprocessing to parallelize oversampling methods
    with multiprocessing.Pool(processes=len(methods)) as pool:
        results = pool.starmap(oversample_method, [(method, samples, regressor) for method in methods])

    return results

def process_oversampling_method(method: Dict, opt_samples_results: pd.DataFrame, scs_regressor, scs_ground_truth_regressor, test_config_set: pd.DataFrame, mc_test_config_set: pd.DataFrame) -> Dict:
    logger.debug(f"Oversampling via {method['method']}")
    new_samples = method["samples"]

    # Generate MC results
    new_samples_results = mc_results_from_configs(new_samples, scs_ground_truth_regressor)
    oversampling = pd.concat([new_samples_results, opt_samples_results], ignore_index=True)

    # Retrain the regressor
    X = oversampling.drop(columns=["SCS"])
    y = oversampling["SCS"]
    scs_regressor.fit(X, y)

    # Run NSGA-II
    new_opt_configs_results = opt_optimization(test_config_set.drop(columns=["SCS"]), scs_regressor)
    # Validation
    _, success_percentage = validate_configurations(new_opt_configs_results, mc_test_config_set)
    logger.debug(f"{method['method']} oversampling concluded: success percentage: {success_percentage}")
    return {"method": method["method"], "success_percentage": success_percentage}

if __name__ == "__main__":
    if(len(sys.argv) > 2):
        n_data_to_verify = int(sys.argv[1])
        n_samples = int(sys.argv[2])
    elif(len(sys.argv) > 1):
        n_data_to_verify = int(sys.argv[1])
        n_samples = 25
    else:
        n_data_to_verify = 200
        n_samples = 25
    initial_dataset = categorical_to_numeric(pd.read_csv("datasets/initial_configurations_to_improve.csv")).drop(columns=["PRSCS_LB", "PRSCS_UB", "FTG_HUM_1", "FTG_HUM_1_LB", "FTG_HUM_1_UB", "FTG_HUM_2", "FTG_HUM_2_LB", "FTG_HUM_2_UB"])
    opt_samples_results, opt_testing_results = train_test_split(initial_dataset, test_size=0.2)
    opt_samples_results = opt_samples_results.sample(n=n_data_to_verify) if len(opt_samples_results) > n_samples else opt_samples_results
    scs_regressor = joblib.load("self_repair/regressor/regressor_SCS_LIME_100.joblib")
    opt_results = opt_optimization(opt_samples_results.drop(columns=["SCS"]), scs_regressor)
    opt_configs = opt_results.drop(columns=["SCS"])
    scs_ground_truth_regressor = joblib.load("self_repair/regressor/regressor_SCS.joblib")
    ground_truth_results = mc_results_from_configs(opt_configs, scs_ground_truth_regressor)
    ground_truth_test_results = mc_results_from_configs(opt_testing_results.drop(columns=["SCS"]), scs_ground_truth_regressor)
    # Calculate accuracy
    stats = []
    invalid_configs, success_percentage = validate_configurations(opt_results, ground_truth_results)
    logger.debug(f"Success percentage before oversampling: {success_percentage}")
    stats.append({"method": "no oversampling", "success_percentage": success_percentage})

    # Oversampling methods
    methods_samples = oversampling_methods_parallel(invalid_configs, n_samples, scs_regressor)

    # Create args list for parallel processing of oversampling methods
    args = [(method, opt_samples_results, scs_regressor, scs_ground_truth_regressor, opt_testing_results, ground_truth_test_results) for method in methods_samples]

    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()

    # Use multiprocessing to process each oversampling method in parallel
    with multiprocessing.Pool(processes=len(methods_samples)) as pool:
        results = pool.starmap(process_oversampling_method, args)

    end_time = time.time()
    logger.debug("Oversampling and validation process completed")
    logger.debug(f"Parallel processing took {end_time - start_time:.2f} seconds")

    # Add results to stats 
    stats.extend(results)

    # Log the oversampling summary
    logger.info("\nOversampling Summary:")
    for stat in stats:
        logger.info(f"{stat['method']:20} -> Success Rate: {stat['success_percentage']:.2f}")

    # Find the best oversampling method based on success percentage
    best_method = max(stats, key=lambda x: x['success_percentage'])
    logger.debug(f"Best resampling method is: {best_method['method']} YUPPY!")