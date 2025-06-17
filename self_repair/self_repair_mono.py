import json
import time
import pandas as pd
import multiprocessing

import self_repair.mc_opt_interface as mc
from self_repair.pipeline import Pipeline
from self_repair.stats import Stat
from self_repair.mc_opt_interface import MC_OPT_INTERFACE
from utils.rp_logger import logger

with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))


# This function performs three main tasks:
# 1. It retrains the regressor with new samples (with ground truth data)
# 2. It optimizes the configurations using the retrained regressor
# 3. It validates the optimized configurations against the ground truth
def oversample_retraing_validation(interface: MC_OPT_INTERFACE, pipeline: Pipeline, target_original, neighbours_original, oversampling_method_name: str, new_samples: pd.DataFrame):
    new_samples = new_samples.drop(columns=["SCS"]) if "SCS" in new_samples.columns else new_samples

    new_samples_results = interface.mc_results_from_configs(new_samples)
    new_regressor = pipeline.retrain_regressor(new_samples_results)

    new_opt_configs_results = interface.opt_optimization(target_original, new_regressor)
    new_groundtruth = interface.mc_results_from_configs(new_opt_configs_results.drop(columns=["SCS"]))
    _, epsilon_array = pipeline.validate_configurations(new_opt_configs_results, new_groundtruth)

    target_neighbours_opt = interface.opt_optimization(neighbours_original, new_regressor)
    new_data_SCS = interface.mc_results_from_configs(target_neighbours_opt.drop(columns=["SCS"]))
    _, neighbours_array = pipeline.validate_configurations(target_neighbours_opt, new_data_SCS)

    return Stat(oversampling_method_name, epsilon_array), Stat(f"{oversampling_method_name}_neighbours", neighbours_array)


def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation: str, points_regressor, max_iterations=1):
    p = Pipeline(
        training_dataset_path="data/dataset1000.csv",
        test_data_path="data/initial_configurations_to_improve.csv",
        points_regressor=points_regressor,
        n_data_to_verify=n_data_to_verify
    )

    mc_opt_interface = mc.RegressorInterface(p.ground_truth_regressor)

    opt_configs = mc_opt_interface.opt_optimization(p.test_set, p.regressor)
    ground_truth_first_test = mc_opt_interface.mc_results_from_configs(opt_configs.drop(columns=["SCS"]))
    invalid_configs, epsilon_array = p.validate_configurations(opt_configs, ground_truth_first_test)

    all_stats = []
    logger.debug(f"Found {len(invalid_configs)} invalid configurations")

    for idx, target in invalid_configs.iterrows():
        if idx >= max_iterations:
            break
        target_df = pd.DataFrame([target])
        target_original_value = p.test_set.iloc[[idx]]
        logger.info(f"Processing invalid configuration index {idx}")

        epsilon_target = epsilon_array[invalid_configs.index.get_loc(idx)]
        oversampling = p.oversample(n_samples, target_df)
        neighbours = p.generate_neighbours_from_config(target_df.drop(columns=["SCS"]), neighbours_to_generate=20)

        neighbours_optimized = mc_opt_interface.opt_optimization(neighbours, p.regressor)
        neighbours_validation = mc_opt_interface.mc_results_from_configs(neighbours_optimized.drop(columns=["SCS"]))
        _, neighbours_array = p.validate_configurations(neighbours_optimized, neighbours_validation)

        all_stats.append(Stat("target", [epsilon_target]))
        all_stats.append(Stat(f"target-{len(neighbours_array)}-neighbours", neighbours_array))

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

        with multiprocessing.Pool(processes=len(args)) as pool:
            parallel_stats = pool.starmap(oversample_retraing_validation, args)

        for stat in parallel_stats:
            all_stats.extend(stat)

    logger.info("Collected Stats Summary:")
    for stat in all_stats:
        logger.info(
            f"Method: {stat.get_method_name()}, "
            f"Epsilon Count: {len(stat.get_epsilon_points())}, "
            f"Mean Epsilon: {sum(stat.get_epsilon_points()) / len(stat.get_epsilon_points()):.4f}"
        )

    return all_stats