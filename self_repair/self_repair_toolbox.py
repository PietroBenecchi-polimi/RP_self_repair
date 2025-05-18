import pandas as pd
from self_repair.oversampling import lime_based_resampling, random_oversampling, smote_oversampling, kde_based_resampling
from self_repair.configuration_validation import validate_configurations
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import multiprocessing
from typing import Dict
import numpy as np
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger
from self_repair.mc_opt_interface import *

def oversample_method(method_name: str, invalid_configs: pd.DataFrame, regressor=None, n_samples: int = 100, target_config = None) -> Dict:
    invalid_configs = invalid_configs.sample(
    n=min(len(invalid_configs), n_samples),
    random_state=128
)
    try:
        if method_name == "Random":
            new_samples = random_oversampling(df=target_config, n_samples=n_samples)
            new_samples["SCS"] = 0
        elif method_name == "Smote":
            new_samples = smote_oversampling(df=invalid_configs)
        elif method_name == "LIME_gaussian":
            new_samples = lime_based_resampling(df=target_config, regressor=regressor, n_samples=n_samples)
            new_samples["SCS"] = 0
        elif method_name == "KDE":
            new_samples = kde_based_resampling(df=target_config, n_samples=n_samples)
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

def oversampling_methods_parallel(invalid_configs: pd.DataFrame, n_samples=100, regressor=None, target_config = None) -> Dict:
    # Create list of oversampling methods
    methods = ["Smote", "Random","LIME_gaussian", "KDE"]
    
    # Use multiprocessing to parallelize oversampling methods
    with multiprocessing.Pool(processes=len(methods)) as pool:
        results = pool.starmap(
            oversample_method,
            [
                (method, invalid_configs, regressor, n_samples, target_config)
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
    return {"method": method["method"], "epsilon": np.mean(epsilon_array), "epsilon_array": epsilon_array}

def train_new_regressor(training_set: pd.DataFrame):
    X_train = training_set.drop(columns=["SCS"])
    y_train = training_set["SCS"]

    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    logger.debug("New regressor trained successfully")
    return regressor