import self_repair.mc_opt_interface as mc

import json
import pandas as pd
from self_repair.pipeline import Pipeline
from self_repair.stats import Stat
from self_repair.mc_opt_interface import MC_OPT_INTERFACE

# imports 
from utils.rp_logger import logger
# paralleling imports
import multiprocessing
import time

with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

# This function performs three main tasks:
# 1. It retrains the regressor with new samples (with ground truth data)
# 2. It optimizes the configurations using the retrained regressor
# 3. It validates the optimized configurations against the ground truth
def oversample_retraing_validation(interface: MC_OPT_INTERFACE, pipeline: Pipeline, target_original, oversampling_method_name: str, new_samples: pd.DataFrame):
    new_samples = new_samples.drop(columns=["SCS"]) if "SCS" in new_samples.columns else new_samples

    new_samples_results = interface.mc_results_from_configs(new_samples)
    # Retrain with train data(given previously) + new samples
    new_regressor = pipeline.retrain_regressor(new_samples_results)

    # optimization + validation
    new_opt_configs_results = interface.opt_optimization(target_original, new_regressor, f"{oversampling_method_name}_{len(new_samples)}")
    # get ground truth
    new_groundtruth = interface.mc_results_from_configs(new_opt_configs_results.drop(columns=["SCS"]))
    _, epsilon_array = pipeline.validate_configurations(new_opt_configs_results, new_groundtruth)

    target_neighbours = pipeline.generate_neighbours_from_config(target_original.drop(columns=["SCS"]), neighbours_to_generate = 20)
    target_neighbours_opt = interface.opt_optimization(target_neighbours, new_regressor, f"{oversampling_method_name}_neighbours")
    new_data_SCS = interface.mc_results_from_configs(target_neighbours_opt.drop(columns=["SCS"]))
    _, neighbours_array = pipeline.validate_configurations(target_neighbours_opt, new_data_SCS)

    return Stat(oversampling_method_name, epsilon_array), Stat(f"{oversampling_method_name}_neighbours", neighbours_array)

def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation: str, points_regressor, skip_cache = True):    
    # Pipeline has 
    # - ground_truth_regressor: regressor trained on the full dataset
    # - regressor: regressor trained on a sample of the dataset
    # - test_set: dataset used for the validation(len = n_data_to_verify)
    p = Pipeline(training_dataset_path = "data/dataset1000.csv",
                           test_data_path = "data/initial_configurations_to_improve.csv", points_regressor = points_regressor, n_data_to_verify = n_data_to_verify)

    mc_opt_interface = mc.RegressorInterface(p.ground_truth_regressor, skip_cache)

    # Comments --> to refactor
    opt_configs = mc_opt_interface.opt_optimization(p.test_set, p.regressor, f"regressor_{points_regressor}")
    ground_truth_first_test = mc_opt_interface.mc_results_from_configs(opt_configs.drop(columns=["SCS"]))
    invalid_configs, epsilon_array = p.validate_configurations(opt_configs, ground_truth_first_test)
    target = invalid_configs.iloc[[0]]
    epsilon_target = epsilon_array[0]
    target_original_value = p.test_set.iloc[[target.index[0]]]
    # contains <method_name, generated_data> without SCS column
    oversampling = p.oversample(n_samples, target)

    target_neighbours = p.generate_neighbours_from_config(target.drop(columns=["SCS"]), neighbours_to_generate = 20)
    new_data_SCS = mc_opt_interface.mc_results_from_configs(target_neighbours.drop(columns=["SCS"]))
    _, neighbours_array = p.validate_configurations(target_neighbours, new_data_SCS)

    # Just add the first validation stats
    initial_stats = [
        Stat("target", [epsilon_target]),  # Stats requires list, not a single float/value
        Stat(f"target-{len(neighbours_array)}-neighbours", neighbours_array)
    ]

    args = [
        (
            mc_opt_interface,
            p,
            target_original_value,
            method,
            oversampling[method]
        ) for method in oversampling.keys()
    ]

    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()

    with multiprocessing.Pool(processes=len(oversampling.keys())) as pool:
        parallel_stats = pool.starmap(oversample_retraing_validation, args)

    end_time = time.time()
    logger.debug(f"Oversampling and validation completed in {(end_time - start_time):.2f} seconds")

    all_stats = initial_stats + [stat for pair in parallel_stats for stat in pair]  # Flatten the list of tuples
    return all_stats