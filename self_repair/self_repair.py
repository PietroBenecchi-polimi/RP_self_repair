import json
import pickle
import os
import pandas as pd
import multiprocessing
import numpy as np

import self_repair.mc_opt_interface as mc
from self_repair.mc_opt_interface import MC_OPT_INTERFACE

from self_repair.pipeline import Pipeline
from self_repair.stats import Stat
from utils.rp_logger import logger

with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

# This function performs three main tasks:
# 1. It retrains the regressor with new samples (with ground truth data)
# 2. It optimizes the configurations using the retrained regressor
# 3. It validates the optimized configurations against the ground truth
def oversample_retraing_validation(interface: MC_OPT_INTERFACE, pipeline: Pipeline, target_original, neighbours_original, oversampling_method_name: str, new_samples: pd.DataFrame):
    new_samples = new_samples.drop(columns=["SCS"]) if "SCS" in new_samples.columns else new_samples

    new_samples_results = interface.mc_results_from_configs(new_samples).reset_index(drop=True)
    new_regressor = pipeline.retrain_regressor(new_samples_results)

    new_opt_configs_results = interface.opt_optimization(target_original, new_regressor).reset_index(drop=True)
    new_groundtruth = interface.mc_results_from_configs(new_opt_configs_results.drop(columns=["SCS"]))
    _, epsilon_array = pipeline.validate_configurations(new_opt_configs_results, new_groundtruth)

    target_neighbours_opt = interface.opt_optimization(neighbours_original, new_regressor).reset_index(drop=True)
    new_data_SCS = interface.mc_results_from_configs(target_neighbours_opt.drop(columns=["SCS"]))
    _, neighbours_array = pipeline.validate_configurations(target_neighbours_opt, new_data_SCS)

    return Stat(oversampling_method_name, epsilon_array, target_neighbours_opt['SCS'], new_data_SCS), Stat(f"{oversampling_method_name}_neighbours", neighbours_array, target_neighbours_opt, new_data_SCS)


def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation: str, points_regressor, max_iterations=1):
    p = Pipeline(
        training_dataset_path="data/dataset1000.csv",
        test_data_path="data/initial_configurations_to_improve.csv",
        points_regressor=points_regressor,
        n_data_to_verify=n_data_to_verify
    )

    # altra versione in codice commentato 
    # mc_opt_interface = mc.ModelCheckerIntercafe()

    mc_opt_interface = mc.RegressorInterface(p.ground_truth_regressor)

    opt_configs = mc_opt_interface.opt_optimization(p.test_set, p.regressor).reset_index(drop=True)
    ground_truth_first_test = mc_opt_interface.mc_results_from_configs(opt_configs.drop(columns=["SCS"]))
    invalid_configs, epsilon_array = p.validate_configurations(opt_configs, ground_truth_first_test)

    all_stats = []

    logger.debug(f"Found {len(invalid_configs)} invalid configurations")

    # Ensure cache directory exists
    os.makedirs("cache", exist_ok=True)

    target_epsilons = {}
    for i in range(len(invalid_configs.index)):
        target_epsilons[i] = epsilon_array[invalid_configs.index[i]]

    epsilon_array = target_epsilons
    invalid_configs = invalid_configs.reset_index(drop=True)

    for idx, target in invalid_configs.iterrows():
        if idx >= max_iterations:
            break
        
        target_df = pd.DataFrame([target])
        target_original_value = p.test_set.iloc[[idx]]
        logger.info(f"Processing invalid configuration index {idx}")
        logger.info("Target configuration:")
        logger.info(f"{target_original_value.to_dict(orient='records')[0]}")
        epsilon_target = epsilon_array[invalid_configs.index.get_loc(idx)]
        neighbours = p.generate_neighbours_from_config(target_df.drop(columns=["SCS"]), neighbours_to_generate=20)

        logger.info("Generated neighbours with variance over all features of:")
        for feature in neighbours.columns:
            logger.info(f"{feature}: {np.var(neighbours[feature])}")
            
        neighbours_optimized = mc_opt_interface.opt_optimization(neighbours, p.regressor)
        neighbours_validation = mc_opt_interface.mc_results_from_configs(neighbours_optimized.drop(columns=["SCS"]))
        _, neighbours_array = p.validate_configurations(neighbours_optimized, neighbours_validation)

        all_stats.append(Stat("target", [epsilon_target], neighbours_optimized["SCS"].to_list(), neighbours_validation["SCS"].to_list()))
        all_stats.append(Stat(f"target-{len(neighbours_array)}-neighbours", neighbours_array, neighbours_optimized["SCS"].to_list(), neighbours_validation["SCS"].to_list()))

        oversampling = p.oversample(n_samples, target_df)

        for method in oversampling.keys():
            logger.info(f"Method: {method}")
            for feature in oversampling[method].columns:
                logger.info(f"{feature}: {oversampling[method][feature].var()}")

        args = [
            (
                mc_opt_interface,
                p,
                target_original_value,
                neighbours,
                method,
                oversampling[method]
            ) for method in oversampling.keys()
        ]

        with multiprocessing.Pool() as pool:
            parallel_stats = pool.starmap(oversample_retraing_validation, args)
            
        logger.info(all_stats[len(all_stats) - 1].__repr__())
        logger.info(all_stats[len(all_stats) - 2].__repr__())
        for stat in parallel_stats:
            all_stats.extend(stat)

        # Save after each iteration
        cache_path = f"cache/all_stats_iter_{idx}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(all_stats, f)
        logger.info(f"Saved all_stats cache to {cache_path}")
        logger.info(all_stats[0].__repr__())
        logger.info(all_stats[1].__repr__())
        for stat in parallel_stats:
            logger.info(stat.__repr__())

    logger.info("Collected Stats Summary:")
    for stat in all_stats:
        logger.info(stat.__repr__())
    logger.info(f"Deleting cache directory")
    if os.path.exists("cache"):
        for file in os.listdir("cache"):
            file_path = os.path.join("cache", file)
            try:
                os.remove(file_path)
                logger.info(f"Deleted cache file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting cache file {file_path}: {e}")

    return all_stats