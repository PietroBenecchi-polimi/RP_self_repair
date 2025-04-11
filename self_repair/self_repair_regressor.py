import pandas as pd
from oversampling import lime_based_resampling, random_oversampling, smote_oversampling
from validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from typing import Dict
import multiprocessing
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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
        "samples": new_samples.drop(columns=["SCS"]),
        "SCS": new_samples["SCS"]
    }

def oversampling_methods_parallel(invalid_configs: pd.DataFrame, n_samples=100, regressor=None) -> Dict:
    # Samples from invalid configurations
    samples = invalid_configs.sample(n=n_samples, random_state=42) if len(invalid_configs) > n_samples else invalid_configs
    # Create list of oversampling methods
    methods = ["Smote", "Random", "LIME_gaussian"]
    
    # Use multiprocessing to parallelize oversampling methods
    with multiprocessing.Pool(processes=len(methods)) as pool:
        results = pool.starmap(oversample_method, [(method, samples, regressor) for method in methods])

    return results

def process_oversampling_method(method: Dict, opt_samples_results: pd.DataFrame, regressor, ground_truth_regressor, test_dataset: pd.DataFrame) -> Dict:
    logger.debug(f"Oversampling via {method['method']}")
    new_samples = method["samples"]

    # Generate MC results
    if(method["method"] == "SMOTE"):
        new_samples_results = pd.concat([method["SCS"], new_samples], ignore_index=True)
    else:
        new_samples_results = mc_results_from_configs(new_samples, ground_truth_regressor)
    oversampling = pd.concat([new_samples_results, opt_samples_results], ignore_index=True)

    # Retrain the regressor
    X = oversampling.drop(columns=["SCS"])
    y = oversampling["SCS"]
    regressor.fit(X, y)

    # Run NSGA-II
    new_opt_configs_results = opt_optimization(test_dataset,regressor, method['method'])
    # Ground truth
    new_groundtruth = mc_results_from_configs(new_opt_configs_results.drop(columns=["SCS"]), ground_truth_regressor)
    # Validation
    _, success_percentage = validate_configurations(new_opt_configs_results, new_groundtruth)
    logger.debug(f"{method['method']} oversampling concluded: success percentage: {success_percentage}")

    return {"method": method["method"], "success_percentage": success_percentage}


def train_new_regressor(training_set: pd.DataFrame, n_points):
    X_train = training_set.drop(columns=["SCS"])
    y_train = training_set["SCS"]

    regressor = RandomForestRegressor(n_estimators=n_points, random_state=42)
    regressor.fit(X_train, y_train)

    logger.debug("New regressor trained successfully")
    return regressor

if __name__ == "__main__":
    n_data_to_verify = 200
    n_samples = 25
    final_validation_invalid_configs = False
    if(len(sys.argv) > 3):
        n_data_to_verify = int(sys.argv[1])
        n_samples = int(sys.argv[2])
        final_validation_invalid_configs = bool(sys.argv[3])
    if(len(sys.argv) > 2):
        n_data_to_verify = int(sys.argv[1])
        n_samples = int(sys.argv[2])
    elif(len(sys.argv) > 1):
        n_data_to_verify = int(sys.argv[1])
    else:
        logger.info(f"No argument passed. Default configuration will be used.")
    logger.info(f"Configuration: datset_size:{n_data_to_verify}, oversampling size:{n_samples}, {'invalid configuration for validation' if final_validation_invalid_configs else 'new dataset for validation'}")
    n_points = 10
    dataset = categorical_to_numeric(pd.read_csv("datasets/dataset1000.csv")).drop(columns=["FTG_HUM_1", "FTG_HUM_1_LB", "FTG_HUM_1_UB", "FTG_HUM_2", "FTG_HUM_2_LB", "FTG_HUM_2_UB"])
    dataset["SCS"] = (dataset["PRSCS_LB"] + dataset["PRSCS_UB"]) /2
    dataset = dataset.drop(columns=["PRSCS_LB", "PRSCS_UB"])
    dataset = dataset.sample(n_points, random_state=42).reset_index(drop=True)
    # Sets up ground truth
    ground_truth_regressor = train_new_regressor(dataset, 1000)
    # Loads dataset and creates regressor on a fixed number of samples
    logger.debug(f"Training new regressor with {n_points} points")
    regressor = train_new_regressor(dataset, n_points)
    # Creates datasets for the before and after oversampling verification
    verification_dataset = categorical_to_numeric(pd.read_csv("datasets/initial_configurations_to_improve.csv")).drop(columns=["FTG_HUM_1", "FTG_HUM_1_LB", "FTG_HUM_1_UB", "FTG_HUM_2", "FTG_HUM_2_LB", "FTG_HUM_2_UB"])
    verification_dataset["SCS"] = (verification_dataset["PRSCS_LB"] + verification_dataset["PRSCS_UB"]) /2
    verification_dataset = verification_dataset.drop(columns=["PRSCS_LB", "PRSCS_UB"])
    first_verification, second_verification = train_test_split(verification_dataset, test_size=0.5, random_state=42)
    first_verification = pd.DataFrame(first_verification).sample(n=n_data_to_verify) if len(first_verification) > n_samples else pd.DataFrame(first_verification)
    second_verification = pd.DataFrame(second_verification).sample(n=n_data_to_verify) if len(second_verification) > n_samples else pd.DataFrame(second_verification)
    # Before resampling verification
    opt_results = opt_optimization(first_verification,regressor, "regressor_100")
    opt_configs = opt_results.drop(columns=["SCS"])
    ground_truth_first_verificatation = mc_results_from_configs(opt_configs, ground_truth_regressor)
    # Calculate accuracy
    stats = []
    invalid_configs, success_percentage = validate_configurations(opt_results, ground_truth_first_verificatation)
    logger.debug(f"Success percentage before oversampling: {success_percentage}")
    stats.append({"method": "no oversampling", "success_percentage": success_percentage})
    second_test = invalid_configs if final_validation_invalid_configs else second_verification
    # Oversampling methods
    methods_samples = oversampling_methods_parallel(invalid_configs, n_samples, regressor)
    # Adds previous regressor without sampling
    # TO DO
    # Create args list for parallel processing of oversampling methods
    args = [(method, dataset, regressor, ground_truth_regressor, second_test) for method in methods_samples]

    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()

    # Use multiprocessing to process each oversampling method in parallel
    with multiprocessing.Pool(processes=len(methods_samples)) as pool:
        results = pool.starmap(process_oversampling_method, args)
    # Verify no resampling on new dataset
    second_validation_results = opt_optimization(second_test, regressor, "invalid_configs_validation" if final_validation_invalid_configs else "second_validation")
    ground_truth_second_verificatation = mc_results_from_configs(second_validation_results.drop(columns=["SCS"]), ground_truth_regressor)
    invalid_configs, success_percentage = validate_configurations(second_validation_results, ground_truth_second_verificatation)
    stats.append({"method": "no oversampling - second verification", "success_percentage": success_percentage})
    #END
    end_time = time.time()
    logger.debug("Oversampling and validation process completed")
    logger.debug(f"Parallel processing took {end_time - start_time:.2f} seconds")

    # Add results to stats 
    stats.extend(results)

    # Log the oversampling summary
    logger.info("\nOversampling Summary:")
    for stat in stats:
        logger.info(f"{stat['method']:40} -> Success Rate: {stat['success_percentage']:.2f}")

    # Find the best oversampling method based on success percentage
    stats.pop(0)
    best_method = max(stats, key=lambda x: x['success_percentage'])
    logger.debug(f"Best resampling method is: {best_method['method']} YUPPY!")