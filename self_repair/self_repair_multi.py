import pandas as pd
from self_repair.oversampling import lime_based_resampling, random_oversampling, smote_oversampling, kde_based_resampling
from self_repair.configuration_validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from typing import Dict
import multiprocessing
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import numpy as np
import utils.datacleaner as ut

warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger
from self_repair.mc_opt_interface import *

def oversample_method(method_name: str, invalid_configs: pd.DataFrame, regressor=None, previous_configs: pd.DataFrame = None, n_samples: int = 100) -> Dict:
    try:
        if method_name == "Random":
            new_samples = random_oversampling(df=invalid_configs, n_samples=n_samples)
            new_samples["SCS"] = 0
        elif method_name == "Smote":
            new_samples = smote_oversampling(df=invalid_configs)
        elif method_name == "Smote-2":
            new_samples = smote_oversampling(df=previous_configs)
        elif method_name == "LIME_gaussian":
            new_samples = lime_based_resampling(df=invalid_configs, regressor=regressor, n_samples=n_samples)
            new_samples["SCS"] = 0
        elif method_name == "KDE":
            new_samples = kde_based_resampling(df=invalid_configs, n_samples=n_samples)
            new_samples["SCS"] = 0
        else:
            raise ValueError(f"Unknown oversampling method: {method_name}")
        return {
            "method": method_name,
            "samples": new_samples.drop(columns=["SCS"]),
            "error": False,
            "SCS": new_samples["SCS"]
        }
    except ValueError as e:
        logger.error(f"Error in oversample_method [{method_name}]: {str(e)}")
        return {"method": method_name, "error": True, "message": str(e)}

def oversampling_methods_parallel(invalid_configs: pd.DataFrame, previous_configs: pd.DataFrame = None, n_samples=100, regressor=None) -> Dict:
    # Create list of oversampling methods
    methods = ["Smote", "Smote-2", "Random","LIME_gaussian", "KDE"]
    
    # Use multiprocessing to parallelize oversampling methods
    with multiprocessing.Pool(processes=len(methods)) as pool:
        results = pool.starmap(
            oversample_method,
            [
                (method, invalid_configs, regressor, previous_configs, n_samples) if method == "Smote-2"
                else (method, invalid_configs, regressor)
                for method in methods
            ]
        )
    results = [result for result in results if result["error"] == False]
    return results

# Run oversample method, re-optimizationa and validatio 
def oversample_and_validation(method: Dict, previous_data: pd.DataFrame, regressor, ground_truth_regressor, test_dataset: pd.DataFrame, skip_cache = True) -> Dict:
    logger.debug(f"Oversampling via {method['method']}")
    new_samples = method["samples"]
    
    # Smote doesn't need to pass to mc. We already have y values.
    if method["method"] in ["Smote", "Smote-2"]:
        new_samples_results = pd.concat(
        [method["SCS"].to_frame(), new_samples], 
        axis=1)
    else:
        new_samples_results = mc_results_from_configs(new_samples, ground_truth_regressor)

    # Create new dataset to retrain regressor
    combined_dataset = pd.concat([previous_data, new_samples_results], ignore_index=True)
    X = combined_dataset.drop(columns=["SCS"])
    y = combined_dataset["SCS"]

    #Train new regressor
    regressor_copy = clone(regressor)
    regressor_copy.fit(X, y)

    new_opt_configs_results = opt_optimization(test_dataset, regressor_copy, f"{method['method']}_{len(test_dataset)}", skip_cache)
    new_groundtruth = mc_results_from_configs(new_opt_configs_results.drop(columns=["SCS"]), ground_truth_regressor)

    # Testing configuration after Re-optimization
    _, epsilon_array = validate_configurations(new_opt_configs_results, new_groundtruth)

    logger.debug(f"{method['method']} oversampling concluded: mean epsilon was: {np.mean(epsilon_array)}")
    return {"method": method["method"], "epsilon_array": epsilon_array}

def train_new_regressor(training_set: pd.DataFrame):
    X_train = training_set.drop(columns=["SCS"])
    y_train = training_set["SCS"]

    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    logger.debug("New regressor trained successfully")
    return regressor

def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation, points_regressor, skip_cache = True):
    logger.info(f"Configuration: dataset_size:{n_data_to_verify}, oversampling size:{n_samples}, Second validation is with: {data_type_second_validation}")
    #Training dataset
    data_path = "data/dataset1000.csv"
    dataset = ut.prepare_dataset(data_path)

    # Model checker, ground truth regressor
    ground_truth_regressor = train_new_regressor(dataset)
    logger.debug(f"Training new regressor with {points_regressor} points")
    dataset = dataset.sample(points_regressor, random_state=128).reset_index(drop=True)
    # stupid regressor trained with at least 1/10 of data of gorund truth
    regressor = train_new_regressor(dataset)

    # Transformation rules for dataset
    data_path = "data/initial_configurations_to_improve.csv"
    verification_dataset = ut.prepare_dataset(data_path)

    # Divide dataset for verification
    first_verification, second_verification = train_test_split(verification_dataset, test_size=0.5, random_state=42)
    first_verification = pd.DataFrame(first_verification).sample(n=n_data_to_verify) if len(first_verification) > n_samples else pd.DataFrame(first_verification)
    second_verification = pd.DataFrame(second_verification).sample(n=n_data_to_verify) if len(second_verification) > n_samples else pd.DataFrame(second_verification)

    # First optimization and retrive data with model checker
    opt_results = opt_optimization(first_verification, regressor, f"regressor_{points_regressor}", skip_cache)
    opt_configs = opt_results.drop(columns=["SCS"])
    ground_truth_first_verificatation = mc_results_from_configs(opt_configs, ground_truth_regressor)

    # Test accuracy of optimization
    stats = []
    invalid_configs, epsilon_array = validate_configurations(opt_results, ground_truth_first_verificatation)
    logger.debug(f"Mean epsilon before oversampling: {np.mean(epsilon_array)}")
    logger.debug(f"Number of invalid configurations: {len(invalid_configs)}")
    stats.append({"method": "no oversampling", "epsilon_array": epsilon_array})

    # Second verification:
    # 3 possible configurations: 
    # 1. Use the invalid configurations from the first verification
    # 2. Use the the entire dataset of the first verification
    # 3. Use a brand new dataset 
    if data_type_second_validation == "invalid_configs":
        second_test = invalid_configs 
    elif data_type_second_validation == "first_verification":
        second_test = first_verification
    else:
        # Use the second verification dataset
        second_test = second_verification

    # Run oversampling, re-optimization and validation 
    methods_samples = oversampling_methods_parallel(invalid_configs=invalid_configs, previous_configs=first_verification, n_samples=n_samples, regressor=regressor)
    args = [(method, dataset, regressor, ground_truth_regressor, second_test, skip_cache) for method in methods_samples]
    #Parallel running
    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()
    # Use multiprocessing to parallelize the oversampling, re-optimization and validation process
    with multiprocessing.Pool(processes=len(methods_samples)) as pool:
        results = pool.starmap(oversample_and_validation, args)

    # Second verification. Is not always constant this part? It should be alwqays the same: 0%
    if data_type_second_validation == "invalid_configs":
        # New optimization process: previous data + oversampling one
        second_validation_results = opt_optimization(second_test, regressor, f"invalid_configs_validation_{points_regressor}", skip_cache)
        ground_truth_second_verificatation = mc_results_from_configs(second_validation_results.drop(columns=["SCS"]), ground_truth_regressor)
        _, epsilon_array = validate_configurations(second_validation_results, ground_truth_second_verificatation)
    
    stats.append({"method": "no oversampling - second verification", "epsilon_array": epsilon_array})

    end_time = time.time()
    logger.debug("Oversampling and validation process completed")
    logger.debug(f"Parallel processing took {end_time - start_time:.2f} seconds a.k.a {(end_time - start_time) / 60:.2f} minutes")

    # Statistics about resampling
    stats.extend(results)
    logger.info("\nOversampling Summary:")
    for stat in stats:
        logger.info(f"{stat['method']:40} -> Mean epsilon: {np.mean(stat['epsilon_array']):.2f}")
    return stats