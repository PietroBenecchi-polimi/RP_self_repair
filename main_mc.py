from sklearn.base import clone
from typing import Dict
import pandas as pd
import utils.datacleaner as dc
import model_checker.hri_designtime.src.hmt_factors as hmtf

import pandas as pd
from self_repair.configuration_validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import multiprocessing
import time
from sklearn.model_selection import train_test_split
import numpy as np
import utils as ut
import main 

warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger
from self_repair.mc_opt_interface import *

def process_data(data):
    # Convert categorical columns to numeric
    data = dc.numeric_to_categorical(data)
    # Combine AGE and STAT into a new column
    data['AGE/STAT 1'] = data['HUM_1_AGE'].astype(str) + '/' + data['HUM_1_STA'].astype(str)
    data['AGE/STAT 2'] = data['HUM_2_AGE'].astype(str) + '/' + data['HUM_2_STA'].astype(str)

    data['Position 1'] = data['HUM_1_POS_X'].astype(str) + ', ' + data['HUM_1_POS_Y'].astype(str)
    data['Position 2'] = data['HUM_2_POS_X'].astype(str) + ', ' + data['HUM_2_POS_Y'].astype(str) 

    data.drop(columns=['HUM_1_POS_X', 'HUM_1_POS_Y', 'HUM_2_POS_X', 'HUM_2_POS_Y', 'HUM_1_AGE', 'HUM_2_AGE',
                      'HUM_1_STA', 'HUM_2_STA'], inplace=True) 
    #Predicted columns drop
    #data.drop(columns=['PRSCS_LB', 'PRSCS_UB', 'SCS', 'FTG_HUM_1_LB', 'FTG_HUM_1_UB', 'FTG_HUM_1',
    #                   'FTG_HUM_2_LB', 'FTG_HUM_2_UB', 'FTG_HUM_2'], inplace=True)
    # Extract new columns
    age_stat_1 = data.pop('AGE/STAT 1')
    age_stat_2 = data.pop('AGE/STAT 2')
    vel_1 = data.pop('Position 1')
    vel_2 = data.pop('Position 2')

    # Desidered position
    data.insert(9, 'AGE/STAT 1', age_stat_1)
    data.insert(11, 'AGE/STAT 2', age_stat_2)
    data.insert(12, 'Position 1', vel_1)
    data.insert(13, 'Position 2', vel_2)

    return data

def mc_results_from_configs(data):
    data = process_data(data)
    data = hmtf.run_mc_simulations(data)

    data.to_csv("data/processed_dataset.csv", index=False)
    return data

# Run oversample method, re-optimizationa and validatio 
def oversample_and_validation(method: Dict, previous_data: pd.DataFrame, regressor, test_dataset: pd.DataFrame, skip_cache = True) -> Dict:
    logger.debug(f"Oversampling via {method['method']}")
    new_samples = method["samples"]
    
    # Smote doesn't need to pass to mc. We already have y values.
    if method["method"] in ["Smote", "Smote-2"]:
        new_samples_results = pd.concat(
        [method["SCS"].to_frame(), new_samples], 
        axis=1)
    else:
        new_samples_results = mc_results_from_configs(new_samples)

    # Create new dataset to retrain regressor
    combined_dataset = pd.concat([previous_data, new_samples_results], ignore_index=True)
    X = combined_dataset.drop(columns=["SCS"])
    y = combined_dataset["SCS"]

    #Train new regressor
    regressor_copy = clone(regressor)
    regressor_copy.fit(X, y)

    new_opt_configs_results = opt_optimization(test_dataset, regressor_copy, f"{method['method']}_{len(test_dataset)}", skip_cache)
    new_groundtruth = mc_results_from_configs(new_opt_configs_results)

    # Testing configuration after Re-optimization
    _, epsilon_array = validate_configurations(new_opt_configs_results, new_groundtruth)

    logger.debug(f"{method['method']} oversampling concluded: mean epsilon was: {np.mean(epsilon_array)}")
    return {"method": method["method"], "epsilon_array": epsilon_array}

def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation, points_regressor, skip_cache = True):
    logger.info(f"Configuration: dataset_size:{n_data_to_verify}, oversampling size:{n_samples}, Second validation is with: {data_type_second_validation}")
    #Training dataset
    data_path = "data/configurations_improved_20_20.csv"
    dataset = pd.read_csv(data_path)
    dataset = dataset.drop(columns=["FTG"])

    dataset = dataset.sample(points_regressor, random_state=128).reset_index(drop=True)
    # stupid regressor trained with at least 1/10 of data of ground truth
    regressor = main.train_new_regressor(dataset)

    # Transformation rules for dataset
    data_path = "data/initial_configurations_to_improve.csv"
    verification_dataset = pd.read_csv(data_path)

    # Divide dataset for verification
    first_verification, second_verification = train_test_split(verification_dataset, test_size=0.5, random_state=42)
    first_verification = pd.DataFrame(first_verification).sample(n=n_data_to_verify) if len(first_verification) > n_samples else pd.DataFrame(first_verification)
    second_verification = pd.DataFrame(second_verification).sample(n=n_data_to_verify) if len(second_verification) > n_samples else pd.DataFrame(second_verification)

    # First optimization and retrive data with model checker
    opt_results = opt_optimization(first_verification, regressor, f"regressor_{points_regressor}", skip_cache)
    ground_truth_first_verificatation = mc_results_from_configs(opt_results)

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
    methods_samples = main.oversampling_methods_parallel(invalid_configs=invalid_configs, previous_configs=first_verification, n_samples=n_samples, regressor=regressor)
    args = [(method, dataset, regressor, second_test, skip_cache) for method in methods_samples]
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
        ground_truth_second_verificatation = mc_results_from_configs(second_validation_results)
        _, epsilon_array = validate_configurations(second_validation_results, ground_truth_second_verificatation)
        
    stats.append({"method": "no oversampling - second verification", "epsilon_array": epsilon_array})

    end_time = time.time()
    logger.debug("Oversampling and validation process completed")
    logger.debug(f"Parallel processing took {end_time - start_time:.2f} seconds")

    # Statistics about resampling
    stats.extend(results)
    logger.info("\nOversampling Summary:")
    for stat in stats:
        logger.info(f"{stat['method']:40} -> Mean epsilon: {np.mean(stat['epsilon_array']):.2f}")
    return stats

if __name__ == "__main__":
    run_oversampling_pipeline(100, 100, "invalid_configs", 100, False)