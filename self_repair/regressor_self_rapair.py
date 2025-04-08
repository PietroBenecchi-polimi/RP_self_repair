import pandas as pd
import joblib
from oversampling import lime_based_resampling, random_oversampling, smote_oversampling
from validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import numpy as np
from typing import Dict
import multiprocessing
warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger
from utils.datacleaner import categorical_to_numeric
from mc_opt_interface import *
import time

def oversampling_methods(invalid_config, n_samples=100, regressor=None):
    new_samples_list = []
    # Oversampling methods:
    # 1. Random oversampling
    new_samples_random = random_oversampling(df=invalid_config, n_samples=n_samples)
    new_samples = {
        "method": "Random",
        "samples": new_samples_random
    }
    new_samples_list.append(new_samples)
    #2. SMOTE based oversampling (Needs to output the SCS and FTG interpolation data!!)
    # new_samples_smote = smote_oversampling(df=invalid_config)
    # new_samples = {
    #     "method": "Smote",
    #     "samples": new_samples_smote
    # }
    # new_samples_list.append(new_samples)

    # 3. LIME based oversampling
    new_samples_lime = lime_based_resampling(df=invalid_config, regressor=regressor)
    new_samples_lime['SCS'] = np.random.uniform(
        low=invalid_config['SCS'].min(), 
        high=invalid_config['SCS'].max(), 
        size=new_samples_lime.shape[0]
    )
    new_samples = {
        "method": "LIME_gaussian",
        "samples": new_samples_lime
    }
    new_samples_list.append(new_samples)

    return new_samples_list

def process_oversampling_method(method: Dict, opt_samples_results: pd.DataFrame, scs_regressor, scs_ground_truth_regressor) -> Dict:
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
    new_opt_configs_results = opt_optimization(new_samples_results.drop(columns=["SCS"]), scs_regressor)
    new_opt_configs = new_opt_configs_results.drop(columns=["SCS"])
    # Generate ground truth
    new_ground_truth_results = mc_results_from_configs(new_opt_configs, scs_ground_truth_regressor)

    # Validation
    _, success_percentage = validate_configurations(new_opt_configs_results, new_ground_truth_results)
    logger.debug(f"{method['method']} oversampling concluded: success percentage: {success_percentage}")
    return {"method": method["method"], "success_percentage": success_percentage}

if __name__ == "__main__":
    scs_regressor = joblib.load("self_repair/regressor/regressor_SCS_LIME_100.joblib")
    opt_results = pd.read_csv("datasets/configurations_improved_20_20.csv")
    opt_configs = opt_results.drop(columns=["SCS", "FTG"])
    opt_samples_results = categorical_to_numeric(pd.read_csv("datasets/initial_configurations_to_improve.csv")).drop(columns=["PRSCS_LB", "PRSCS_UB", "FTG_HUM_1", "FTG_HUM_1_LB", "FTG_HUM_1_UB", "FTG_HUM_2", "FTG_HUM_2_LB", "FTG_HUM_2_UB"])
    scs_ground_truth_regressor = joblib.load("self_repair/regressor/regressor_SCS.joblib")
    ground_truth_results = mc_results_from_configs(opt_configs, scs_ground_truth_regressor)
    # Calculate accuracy
    stats = []
    invalid_configs, success_percentage = validate_configurations(opt_results, ground_truth_results)
    invalid_configs = invalid_configs.drop(columns=["FTG"])
    logger.debug(f"Success percentage before oversampling: {success_percentage}")
    stats.append({"method": "base", "success_percentage": success_percentage})
    #Oversampling
    methods_samples = oversampling_methods(invalid_configs, 100, scs_regressor)
    # Create args list
    args = [(method, opt_samples_results, scs_regressor, scs_ground_truth_regressor) for method in methods_samples]
    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()
    with multiprocessing.Pool(processes=len(methods_samples)) as pool:
        results = pool.starmap(process_oversampling_method, args)
    end_time = time.time()
    logger.debug("Oversampling and validation process completed")
    logger.debug(f"Parallel processing took {end_time - start_time:.2f} seconds")
    stats.extend(results)
    overview = "Overview"
    for stat in stats:
        overview = overview + f"\n{stat}"
    logger.debug(overview)
    best_method = max(stats, key=lambda x: x['success_percentage'])
    logger.debug(f"Best resampling method is: {best_method['method']} YUPPY!")
    

