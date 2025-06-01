import pipeline as pipeline
import mc_opt_interface as mc
import json
import pandas as pd
import utils.generateData as gd
from self_repair.pipeline import Pipeline
from self_repair.stats import Stat
from self_repair.mc_opt_interface import MC_OPT_INTERFACE
import utils.generateData as gd

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
def oversample_retraing_validation(interface: MC_OPT_INTERFACE, pipeline: Pipeline, test_dataset, oversampling_method_name: str, new_samples: pd.DataFrame):
    new_samples = new_samples.drop(columns=["SCS"]) if "SCS" in new_samples.columns else new_samples

    new_samples_results = interface.mc_results_from_configs(new_samples)
    # Retrain with train data(given previously) + new samples
    new_regressor = pipeline.retrain_regressor(new_samples_results)

    # optimization + validation
    new_opt_configs_results = interface.opt_optimization(test_dataset, new_regressor, f"{oversampling_method_name}_{len(test_dataset)}")
    # get ground truth
    new_groundtruth = interface.mc_results_from_configs(new_opt_configs_results.drop(columns=["SCS"]), pipeline.ground_truth_regressor)
    _, epsilon_array = pipeline.validate_configurations(new_opt_configs_results, new_groundtruth)

    new_data = gd.generate_neighbours_from_config(test_dataset.iloc[[0]].drop(columns=["SCS"]), pipeline.regressor, neighbours_to_generate = 20)
    new_data_SCS = mc.mc_results_from_configs(new_data.drop(columns=["SCS"]), pipeline.ground_truth_regressor)
    _, neighbours_array = pipeline.validate_configurations(new_data, new_data_SCS)

    return Stat(oversampling_method_name, epsilon_array), Stat(f"{oversampling_method_name}_neighbours", neighbours_array)

def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation: str, points_regressor, skip_cache = True):
    # stats contains stats.py objects >
    stats = []
    
    # Pipeline has 
    # - ground_truth_regressor: regressor trained on the full dataset
    # - regressor: regressor trained on a sample of the dataset
    # - test_set: dataset used for the validation(len = n_data_to_verify)
    p = pipeline.Pipeline(training_dataset_path = "data/dataset1000.csv",
                           test_data_path = "data/initial_configurations_to_improve.csv", points_regressor = points_regressor, n_data_to_verify = n_data_to_verify)

    m = mc.RegressorInterface(p.ground_truth_regressor, skip_cache)

    # Comments --> to refactor
    opt_configs = mc.opt_optimization(p.test_set, p.regressor, f"regressor_{points_regressor}", skip_cache)
    ground_truth_first_test = mc.mc_results_from_configs(opt_configs.drop(columns=["SCS"]), p.ground_truth_regressor)
    invalid_configs, epsilon_array = p.validate_configurations(opt_configs, ground_truth_first_test)
    
    # contains <method_name, generated_data> without SCS column
    generated_data_methods = p.oversample(n_samples, invalid_configs.iloc[[0]])

    new_data = gd.generate_neighbours_from_config(invalid_configs.iloc[[0]].drop(columns=["SCS"]), p.regressor, neighbours_to_generate = 20)
    new_data_SCS = mc.mc_results_from_configs(new_data.drop(columns=["SCS"]), p.ground_truth_regressor)
    _, neighbours_array = p.validate_configurations(opt_configs, new_data_SCS)

    # Just add the first validation stats
    initial_stats = [
        Stat("no oversampling", epsilon_array[0]),
        Stat("neighbours", neighbours_array)
    ]

    args = [
        (
            m,
            pipeline,
            p.test_set.iloc[[0]], # gives a single row DataFrame, not Dataseries
            method,
            generated_data_methods[method]
        ) for method in generated_data_methods.keys()
    ]

    logger.debug("Starting the oversampling and validation process")
    start_time = time.time()

    with multiprocessing.Pool(processes=len(generated_data_methods)) as pool:
        parallel_stats = pool.starmap(oversample_retraing_validation, args)

    end_time = time.time()
    logger.debug(f"Oversampling and validation completed in {(end_time - start_time):.2f} seconds")

    all_stats = initial_stats + parallel_stats
    return all_stats