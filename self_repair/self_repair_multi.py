import pandas as pd
from self_repair.configuration_validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import multiprocessing
import time
from sklearn.model_selection import train_test_split
import numpy as np
import utils.datacleaner as ut
from self_repair.self_repair_toolbox import train_new_regressor, opt_optimization, mc_results_from_configs, oversample_and_validation, oversampling_methods_parallel

warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger
from self_repair.mc_opt_interface import *

def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation, points_regressor, skip_cache = True):
    logger.info(f"Configuration: dataset_size:{n_data_to_verify}, oversampling size:{n_samples}, Second validation is with: {data_type_second_validation}")
    #Training dataset
    data_path = "data/dataset1000.csv"
    dataset = ut.load_dataset_for_regressor(data_path)

    # Model checker, ground truth regressor
    ground_truth_regressor = train_new_regressor(dataset)
    logger.debug(f"Training new regressor with {points_regressor} points")
    dataset = dataset.sample(points_regressor, random_state=128).reset_index(drop=True)
    # stupid regressor trained with at least 1/10 of data of gorund truth
    regressor = train_new_regressor(dataset)

    # Transformation rules for dataset
    data_path = "data/initial_configurations_to_improve.csv"
    verification_dataset = ut.load_dataset_for_regressor(data_path)

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