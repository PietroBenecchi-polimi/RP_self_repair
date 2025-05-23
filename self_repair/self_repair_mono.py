import pandas as pd
from self_repair.configuration_validation import validate_configurations, validate_configuration
import multiprocessing
import time
import numpy as np
import utils.datacleaner as ut
from utils.rp_logger import logger
from self_repair.self_repair_toolbox import train_new_regressor, opt_optimization, mc_results_from_configs, oversample_and_validation, oversampling_methods_parallel
from utils.datacleaner import get_transformation_rules
import json
with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

def synthesize_data(target_config: pd.DataFrame):
    n_samples = 20
    synthetic_data = {}
    transformation_rules = get_transformation_rules()

    for factor_key in target_config.columns:
        if factor_key == "SCS":
            synthetic_data[factor_key] = np.random.uniform(0,1, n_samples)
        elif factor_key not in transformation_rules.keys():
            if "PRGS" == factor_key:
                synthetic_data[factor_key] = np.random.choice([0,1,2,3,4,5])
            else:
                if factor_key in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                    col_max, col_min = factors["HUM_1_POS"]["max_x"], factors["HUM_1_POS"]["min_x"]
                elif factor_key in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                    col_max, col_min = factors["HUM_1_POS"]["max_y"], factors["HUM_1_POS"]["min_y"]
                elif "max" in factors[factor_key]:
                    col_max, col_min = factors[factor_key]["max"], factors[factor_key]["min"]

                synthetic_data[factor_key] = np.random.uniform(col_min, col_max, n_samples)
        else:
            values = list(transformation_rules[factor_key].values())
            synthetic_data[factor_key] = np.random.choice(values, n_samples)

    return pd.DataFrame(synthetic_data)


def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation: str, points_regressor, skip_cache = True):
    logger.info(f"Configuration: dataset_size:{n_data_to_verify}, oversampling size:{n_samples}, Second validation is with: {data_type_second_validation}")
    stats = []
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
    verification_dataset = ut.load_dataset_for_regressor(data_path).sample(n_data_to_verify, random_state=128).reset_index(drop=True)

    # First optimization and retrive data with model checker
    opt_results = opt_optimization(verification_dataset, regressor, f"regressor_{points_regressor}", skip_cache)
    opt_configs = opt_results.drop(columns=["SCS"])
    ground_truth_first_verificatation = mc_results_from_configs(opt_configs, ground_truth_regressor)
    invalid_configs, epsilon_array = validate_configurations(opt_results, ground_truth_first_verificatation)
    stats.append({"method": "regressor - no oversampling", "epsilon": np.mean(epsilon_array), "epsilon_array": epsilon_array})
    # Select target configuration
    for i in range(len(opt_results)):
        verification = ground_truth_first_verificatation.iloc[i]
        result = opt_results.iloc[i]
        try:
            target_config, epsilon = validate_configuration(verification, result)
            break
        except ValueError:
            continue
    target_configs = synthesize_data(target_config=target_config)

    target_configs_verification = mc_results_from_configs(target_configs, ground_truth_regressor)
    invalid_configs, epsilon_array = validate_configurations(target_configs, target_configs_verification)
    logger.debug(f"Epsilon target ball configuration before oversampling: {np.mean(epsilon_array)}")
    stats.append({"method": "target_ball - no oversampling", "epsilon": {np.mean(epsilon_array)}, "epsilon_array": epsilon_array})

    # Second verification:
    if data_type_second_validation == "invalid_configs":
        second_test = target_configs
    else:
        second_test = verification_dataset

    # Second verification
    if data_type_second_validation == "invalid_configs":
        # New optimization process: previous data + oversampling one
        second_validation_results = opt_optimization(second_test, regressor, f"invalid_configs_validation_{points_regressor}", skip_cache)
        ground_truth_second_verificatation = mc_results_from_configs(second_validation_results.drop(columns=["SCS"]), ground_truth_regressor)
        _, epsilon_array = validate_configurations(second_validation_results, ground_truth_second_verificatation)
    
    stats.append({"method": "target_ball - second verification", "epsilon": np.mean(epsilon_array)})
    # Run oversampling, re-optimization and validation
    methods_samples = oversampling_methods_parallel(invalid_configs=invalid_configs, n_samples=n_samples, regressor=regressor, target_config=pd.DataFrame([target_config["config"]]))
    args = [(method, dataset, regressor, ground_truth_regressor, second_test, skip_cache) for method in methods_samples]
    #Parallel running
    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()
    # Use multiprocessing to parallelize the oversampling, re-optimization and validation process
    with multiprocessing.Pool(processes=len(methods_samples)) as pool:
        results = pool.starmap(oversample_and_validation, args)

    end_time = time.time()
    logger.debug("Oversampling and validation process completed")
    logger.debug(f"Parallel processing took {end_time - start_time:.2f} seconds a.k.a {(end_time - start_time) / 60:.2f} minutes")

    # Statistics about resampling
    stats.extend(results)
    logger.info("\nOversampling Summary:")
    for stat in stats:
        logger.info(f"{stat['method']:40} -> Mean epsilon: {stat['epsilon']:.2f}")
    return stats

## USE CASE: 
## Implemts camilli test two.
## The oversampling is done only on unsuccesfful mission so that dataset = dataset[dataset['SCS'] < 0.5]
## This is done with optimized configurations and the SCS is calculated with the regressor
#TODO implement different threshold for SCS and print the graphs together-->ASK LUIGI
def run_oversampling_pipeline_unsuccessfull_mission(n_data_to_verify, n_samples, data_type_second_validation: str, points_regressor, skip_cache = True):
    logger.info(f"Configuration: dataset_size:{n_data_to_verify}, oversampling size:{n_samples}, Second validation is with: {data_type_second_validation}")
    stats = []
    #Training dataset
    data_path = "data/dataset1000.csv"
    dataset = ut.load_dataset_for_regressor(data_path)

    # Model checker, ground truth regressor
    ground_truth_regressor = train_new_regressor(dataset)
    logger.debug(f"Training new regressor with {points_regressor} points")

    # TEST 2: Oversampling on failed mission
    dataset['SCS'] = ground_truth_regressor.predict(dataset)
    dataset = dataset[dataset['SCS'] < 0.5]
    dataset = dataset.sample(points_regressor, random_state=128).reset_index(drop=True)

    # stupid regressor trained with at least 1/10 of data of ground truth
    regressor = train_new_regressor(dataset)

    # Transformation rules for dataset
    data_path = "data/initial_configurations_to_improve.csv"
    verification_dataset = ut.load_dataset_for_regressor(data_path).sample(n_data_to_verify, random_state=128).reset_index(drop=True)

    # First optimization and retrive data with model checker
    opt_results = opt_optimization(verification_dataset, regressor, f"regressor_{points_regressor}", skip_cache)
    opt_configs = opt_results.drop(columns=["SCS"])
    ground_truth_first_verificatation = mc_results_from_configs(opt_configs, ground_truth_regressor)
    invalid_configs, epsilon_array = validate_configurations(opt_results, ground_truth_first_verificatation)
    stats.append({"method": "regressor - no oversampling", "epsilon": np.mean(epsilon_array), "epsilon_array": epsilon_array})
    # Select target configuration
    for i in range(len(opt_results)):
        verification = ground_truth_first_verificatation.iloc[i]
        result = opt_results.iloc[i]
        try:
            target_config, epsilon = validate_configuration(verification, result)
            break
        except ValueError:
            continue
    target_configs = synthesize_data(target_config=target_config)

    target_configs_verification = mc_results_from_configs(target_configs, ground_truth_regressor)
    invalid_configs, epsilon_array = validate_configurations(target_configs, target_configs_verification)
    logger.debug(f"Epsilon target ball configuration before oversampling: {np.mean(epsilon_array)}")
    stats.append({"method": "target_ball - no oversampling", "epsilon": {np.mean(epsilon_array)}, "epsilon_array": epsilon_array})

    # Second verification:
    if data_type_second_validation == "invalid_configs":
        second_test = target_configs
    else:
        second_test = verification_dataset

    # Second verification
    if data_type_second_validation == "invalid_configs":
        # New optimization process: previous data + oversampling one
        second_validation_results = opt_optimization(second_test, regressor, f"invalid_configs_validation_{points_regressor}", skip_cache)
        ground_truth_second_verificatation = mc_results_from_configs(second_validation_results.drop(columns=["SCS"]), ground_truth_regressor)
        _, epsilon_array = validate_configurations(second_validation_results, ground_truth_second_verificatation)
    
    stats.append({"method": "target_ball - second verification", "epsilon": np.mean(epsilon_array)})
    # Run oversampling, re-optimization and validation
    methods_samples = oversampling_methods_parallel(invalid_configs=invalid_configs, n_samples=n_samples, regressor=regressor, target_config=pd.DataFrame([target_config["config"]]))
    args = [(method, dataset, regressor, ground_truth_regressor, second_test, skip_cache) for method in methods_samples]
    #Parallel running
    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()
    # Use multiprocessing to parallelize the oversampling, re-optimization and validation process
    with multiprocessing.Pool(processes=len(methods_samples)) as pool:
        results = pool.starmap(oversample_and_validation, args)

    end_time = time.time()
    logger.debug("Oversampling and validation process completed")
    logger.debug(f"Parallel processing took {end_time - start_time:.2f} seconds a.k.a {(end_time - start_time) / 60:.2f} minutes")

    # Statistics about resampling
    stats.extend(results)
    logger.info("\nOversampling Summary:")
    for stat in stats:
        logger.info(f"{stat['method']:40} -> Mean epsilon: {stat['epsilon']:.2f}")
    return stats

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_oversampling_pipeline(10, 12, "standard_config", 150)