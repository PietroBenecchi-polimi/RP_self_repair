import pandas as pd
import numpy as np
import warnings
import time
import multiprocessing
from sklearn.exceptions import InconsistentVersionWarning

import utils.datacleaner as ut
from utils.rp_logger import logger
from self_repair.mc_opt_interface import (
    opt_optimization,
    mc_results_from_configs
)
from self_repair.pipeline import Pipeline
from self_repair.stats import Stat

warnings.simplefilter("ignore", InconsistentVersionWarning)

def oversampling_validation(pipeline: Pipeline, test_dataset, oversampling_method: str, new_samples: pd.DataFrame, skip_cache = False):
    new_regressor = pipeline.retrain_regressor(new_samples)
    new_opt_configs_results = opt_optimization(test_dataset, new_regressor, f"{oversampling_method}_{len(test_dataset)}", skip_cache)
    new_groundtruth = mc_results_from_configs(new_opt_configs_results.drop(columns=["SCS"]), pipeline.ground_truth_regressor)
    invalid_configs, epsilon_array = pipeline.validate_configurations(new_opt_configs_results, new_groundtruth)
    return Stat(oversampling_method, epsilon_array)

def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation, points_regressor, skip_cache=False) -> list:
    logger.info(
        f"Configuration: dataset_size:{n_data_to_verify}, oversampling size:{n_samples}, "
        f"Second validation type: {data_type_second_validation}"
    )

    # Initialize pipeline
    training_data_path = "data/dataset1000.csv"
    verification_data_path = "data/initial_configurations_to_improve.csv"
    pipeline = Pipeline(training_data_path, verification_data_path, points_regressor, n_data_to_verify)

    # Load and sample verification dataset
    verification_dataset = ut.load_dataset_for_regressor(verification_data_path)
    first_verification = verification_dataset.sample(n=n_data_to_verify) if len(verification_dataset) > n_samples else verification_dataset

    # First optimization and evaluation
    opt_results = opt_optimization(first_verification, pipeline.regressor, f"regressor_{points_regressor}", skip_cache)
    opt_configs = opt_results.drop(columns=["SCS"])
    ground_truth_results = mc_results_from_configs(opt_configs, pipeline.ground_truth_regressor)
    invalid_configs, epsilon_array = pipeline.validate_configurations(opt_results, ground_truth_results)

    logger.debug(f"Mean epsilon before oversampling: {np.mean(epsilon_array):.4f}")
    logger.debug(f"Number of invalid configurations: {len(invalid_configs)}")

    stats = [Stat("no oversampling", epsilon_array)]

    # Determine second test dataset
    second_test = (
        invalid_configs if data_type_second_validation == "invalid_configs"
        else first_verification
    )

    # Prepare oversampling and validation tasks
    methods_samples = pipeline.oversample(n_samples=n_samples, invalid_configurations=invalid_configs)

    args = [
        (
            pipeline,
            second_test,
            method,
            methods_samples[method]
        ) for method in methods_samples.keys()
    ]

    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()

    with multiprocessing.Pool(processes=len(methods_samples)) as pool:
        stats = pool.starmap(oversampling_validation, args)

    end_time = time.time()
    logger.debug(f"Oversampling and validation completed in {(end_time - start_time):.2f} seconds")

    if data_type_second_validation == "invalid_configs":
        second_results = opt_optimization(second_test, pipeline.regressor, f"invalid_configs_validation_{points_regressor}", skip_cache)
        second_ground_truth = mc_results_from_configs(second_results.drop(columns=["SCS"]), pipeline.ground_truth_regressor)
        _, second_epsilons = pipeline.validate_configurations(second_results, second_ground_truth)
        stats.append(Stat("no oversampling - second verification", second_epsilons))

    # Logging final stats
    logger.info("\nOversampling Summary:")
    for stat in stats:
        logger.info(
            f"{stat.get_method_name():40} -> Mean: {stat.get_average_epsilon():.4f}, "
            f"Median: {stat.get_median_epsilon():.4f}, "
            f"Success: {stat.get_n_mission_success()}, "
            f"Failed: {stat.get_n_mission_failed()}"
        )

    return stats