import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import ast
import math
import json
import numpy as np
from scipy.spatial import KDTree

import json

def find_similar_configs(df_A, df_B, epsilon_percentage, original_df=None):
    """
    Finds similar configurations between two datasets (df_A and df_B) based on a given epsilon percentage.

    This function compares configurations in df_B with those in df_A. A configuration in df_B is considered
    similar to a configuration in df_A if the difference between their factor values is within the specified
    epsilon percentage of the range of the factor.

    Args:
        df_A (pd.DataFrame): The reference dataset containing configurations to compare against.
        df_B (pd.DataFrame): The target dataset containing configurations to find matches for.
        epsilon_percentage (float): The percentage of the factor range used to determine similarity.
        original_df (pd.DataFrame, optional): The original dataset to use for mc_config if provided. Defaults to None.

    Returns:
        dict: A dictionary containing the results of the matching process, including:
            - data: A list of dictionaries with matched configurations and their details.
            - epsilon: The epsilon percentage used for matching.
            - method: The method used for matching ("factor_difference").
    """
    with open("./datasets/hmtfactor_config.json", "r") as file:
        factors = json.load(file)
    
    dataset_A = df_A.to_dict(orient="records")
    dataset_B = df_B.to_dict(orient="records")
    results = []
    
    for _, opt_config in enumerate(dataset_A):
        for mc_idx, mc_config in enumerate(dataset_B):
            valid = True
            for factor_key in opt_config.keys():
                if factor_key not in mc_config:
                    valid = False
                    break
                
                difference = abs(opt_config[factor_key] - mc_config[factor_key])
                
                if factor_key in factors:
                    if "max" in factors[factor_key]:
                        range_val = factors[factor_key]["max"] - factors[factor_key]["min"]
                        valid = valid and (difference / range_val < epsilon_percentage)
                    else:
                        valid = valid and (difference < epsilon_percentage)
                elif factor_key in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                    range_val = factors["HUM_1_POS"]["max_x"] - factors["HUM_1_POS"]["min_x"]
                    valid = valid and (difference / range_val < epsilon_percentage)
                elif factor_key in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                    range_val = factors["HUM_1_POS"]["max_y"] - factors["HUM_1_POS"]["min_y"]
                    valid = valid and (difference / range_val < epsilon_percentage)
                else:
                    valid = valid and (difference < epsilon_percentage)
                
                if not valid:
                    break
            if valid:
                results.append({
                    "opt_config": opt_config,
                    "mc_config": (dataset_A[mc_idx] if original_df is None else original_df.iloc[mc_idx].to_dict()),
                    "has_match": True
                })
                break
        if(not valid):
            results.append({
                    "opt_config": opt_config,
                    "mc_config": None,
                    "has_match": False
                })
    
    return {"data": results, "epsilon": epsilon_percentage, "method": "factor_difference"}


def find_similar_configs_old(df_B, df_A, epsilon_percentage, original_df=None):
    """
    Finds similar configurations between two datasets (df_B and df_A) using a KDTree-based approach.

    This function uses a KDTree to find configurations in df_A that are within a proportional epsilon distance
    of configurations in df_B. The epsilon is calculated as a percentage of the norm of the configuration.

    Args:
        df_B (pd.DataFrame): The target dataset containing configurations to find matches for.
        df_A (pd.DataFrame): The reference dataset containing configurations to compare against.
        epsilon_percentage (float): The percentage of the norm used to determine the search radius.
        original_df (pd.DataFrame, optional): The original dataset to use for mc_config if provided. Defaults to None.

    Returns:
        dict: A dictionary containing the results of the matching process, including:
            - data: A list of dictionaries with matched configurations and their details.
            - epsilon: The epsilon percentage used for matching.
            - method: The method used for matching ("norm").
    """
    dataset_A = df_A.to_numpy()
    dataset_B = df_B.to_numpy()
    tree = KDTree(dataset_A)
    results = []
    for i, config in enumerate(dataset_B):
        norm_config = np.linalg.norm(config)
        proportional_epsilon = epsilon_percentage * norm_config
        indices = tree.query_ball_point(config, r=proportional_epsilon)
        if indices:
            closest_idx = min(indices, key=lambda i: np.linalg.norm(dataset_A[i] - config))
            mc_config_dict = original_df.iloc[closest_idx].to_dict() if original_df is not None else df_A.iloc[closest_idx].to_dict()
            results.append({
                "opt_config": df_B.iloc[i].to_dict(),
                "mc_config": mc_config_dict,
                "closest_idx": closest_idx,
                "has_match": True
            })
        else:
            results.append({
                "opt_config": df_B.iloc[i].to_dict(),
                "mc_config": None,
                "closest_idx": None,
                "has_match": False
            })
    return {"data": results, "epsilon": epsilon_percentage, "method": "norm"}


def match_results_analyer(match_results):
    """
    Analyzes the results of configuration matching and saves the differences to a CSV file.

    This function calculates the differences between matched configurations and saves the results
    to a CSV file. It uses factor ranges from a JSON file to normalize differences where applicable.

    Args:
        match_results (dict): The results of the configuration matching process, as returned by
                             `find_similar_configs` or `find_similar_configs_old`.

    Returns:
        None: The function saves the results to a CSV file and prints the output path.
    """
    with open("./datasets/hmtfactor_config.json", "r") as file:
        factors = json.load(file)
    with open("./datasets/metrics.json", "r") as file:
        metrics = json.load(file)
    output_csv_path = f"./Matching_and_verifier/match_results/opt_mc_difference_{match_results['method']}_{match_results['epsilon']}_.csv"
    # Convert DataFrame to a list of dictionaries
    match_results = match_results["data"]
    match_results = [match for match in match_results if match["has_match"] == True]
    print(f"Total matches: {len(match_results)}")
    results = []

    for row in match_results:
        opt_conf = row["opt_config"]
        mc_conf = row["mc_config"]
        row_result = {}
        if not mc_conf:
            continue

        # Calculate differences for each factor key
        for factor_key in mc_conf.keys():
            if(factor_key in metrics.keys()):
                continue
            difference = abs(mc_conf[factor_key] - opt_conf[factor_key])
            if factor_key in factors.keys():
                if("max" in factors[factor_key].keys()):
                    range = factors[factor_key]["max"] - factors[factor_key]["min"]
                    row_result[f"{factor_key}(%)"] = difference / range
                else:
                    row_result[f"{factor_key}"] = difference
            elif factor_key == "HUM_1_POS_X" or factor_key == "HUM_2_POS_X":
                range = factors["HUM_1_POS"]["max_x"] - factors["HUM_1_POS"]["min_x"]
                row_result[f"{factor_key}(%)"] = difference / range
            elif factor_key == "HUM_1_POS_Y" or factor_key == "HUM_2_POS_Y":
                range = factors["HUM_1_POS"]["max_y"] - factors["HUM_1_POS"]["min_y"]
                row_result[f"{factor_key}(%)"] = difference / range
            else:
                row_result[f"{factor_key}"] = difference

        # Add the row result to the results list
        results.append(row_result)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

def validate_configurations(results_to_verify, FTG_threshold=0.01):
    """
    Validates configurations between an optimized dataset and a Model Checker dataset.

    This function takes the results of a configuration matching process (from `find_similar_configs`)
    and validates the SCS and FTG parameters for each matched configuration. It calculates the
    success percentage of valid configurations based on the validation criteria.

    Args:
        results_to_verify (dict): The results of the configuration matching process, as returned by
                                 `find_similar_configs`. It should contain a "data" key with a list
                                 of matched configurations.
        FTG_threshold (float, optional): The threshold for validating the FTG parameter. Defaults to 0.01.

    Returns:
        dict: A dictionary containing the validation results, including:
            - total_comparisons (int): Total number of comparisons made.
            - invalid_count (int): Number of invalid configurations.
            - success_percentage (float): Percentage of valid configurations (rounded to 2 decimal places).
    """
    validity_array = []

    def validate_scs(opt_SCS, mc_SCS):
        """
        Validate the SCS parameter.

        Args:
            opt_SCS (float): The SCS value from the optimized configuration.
            mc_SCS (float): The SCS value from the Model Checker configuration.

        Returns:
            bool: True if the SCS values are valid, False otherwise.
        """
        if opt_SCS > 0.9:
            return bool(mc_SCS)
        else:
            return not bool(mc_SCS)

    def validate_ftg(mc_ftg, opt_ftg, threshold=FTG_threshold):
        """
        Validate the FTG parameter.

        Args:
            mc_ftg (float): The FTG value from the Model Checker configuration.
            opt_ftg (float): The FTG value from the optimized configuration.
            threshold (float, optional): The threshold for validating the FTG parameter. Defaults to FTG_threshold.

        Returns:
            bool: True if the FTG values are valid, False otherwise.
        """
        if mc_ftg == 0 and opt_ftg == 0:
            return True
        else:
            return abs(mc_ftg - opt_ftg) <= threshold

    data_list = []
    # Iterate through the results and validate each configuration
    for result in results_to_verify["data"]:
        if result["has_match"]:
            opt_SCS = result["opt_config"]["SCS"]
            mc_SCS = result["mc_config"]["SCS"]
            validity = validate_scs(opt_SCS, mc_SCS)

            # Validate FTG parameter
            mc_ftg = result["mc_config"]["FTG"]
            opt_ftg = result["opt_config"]["FTG"]
            validity = validity and validate_ftg(mc_ftg, opt_ftg, FTG_threshold)

            if not validity:
                data_list.append({
                    "opt_config": result["opt_config"],
                    "validity": validity
                })
            validity_array.append(validity)
    
    df = pd.DataFrame(data_list)
    df.to_csv(f"./Matching_and_verifier/verifier_results/invalid_configs.csv")
    # Calculate validation metrics
    invalid_count = sum(not v for v in validity_array)
    if(len(validity_array) == 0):
        success_percentage = 0
    else:
        success_percentage = round(1 - (invalid_count / len(validity_array)), 2)

    return {
        "total_comparisons": len(validity_array),
        "invalid_count": invalid_count,
        "success_percentage": success_percentage
    }