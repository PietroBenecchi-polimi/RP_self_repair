"""
This script compares two datasets (optimized_set and dataset_A) to find similar configurations within a user-defined proportional epsilon range.

Usage: Input the two datasets and a given epsilon (between 0 and 1), and the function will provide an array of both optimized configuration and MC configuration, along with whether they match or not.
"""
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
    # Load factors from JSON file
    with open("./datasets/hmtfactor_config.json", "r") as file:
        factors = json.load(file)
    
    # Convert dataframes to dictionaries
    dataset_A = df_A.to_dict(orient="records")
    dataset_B = df_B.to_dict(orient="records")
    results = []
    
    for opt_idx, opt_config in enumerate(dataset_B):
        for mc_idx, mc_config in enumerate(dataset_A):
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
    
    return results

def find_similar_configs_old(df_B, df_A, epsilon_percentage, original_df=None):
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
            # Use original_df if provided, otherwise use df_A
            mc_config_dict = original_df.iloc[closest_idx].to_dict() if original_df is not None else df_A.iloc[closest_idx].to_dict()
            results.append({
                "opt_config": df_B.iloc[i].to_dict(),  # Query configuration with column names
                "mc_config": mc_config_dict,           # Matched configuration with column names
                "closest_idx": closest_idx,            # Index of the closest match
                "has_match": True                     # Match status
            })
        else:
            results.append({
                "opt_config": df_B.iloc[i].to_dict(),  # Query configuration with column names
                "mc_config": None,                     # No match found
                "closest_idx": None,                  # No index
                "has_match": False                   # Match status
            })
    return results

def match_results_analyer(match_results): 
    with open("./datasets/hmtfactor_config.json", "r") as file:
        factors = json.load(file)
    with open("./datasets/metrics.json", "r") as file:
        metrics = json.load(file)
    output_csv_path = "./ImprovedDataToMCMatches/match_results/opt_mc_difference.csv"
    # Convert DataFrame to a list of dictionaries
    match_results = match_results.to_dict(orient='records')
    match_results = [match for match in match_results if match["Has_Match"] == True]
    print(f"Total matches {len(match_results)}")
    results = []

    for row in match_results:
        try:
            opt_conf = ast.literal_eval(row.get("opt_config", "{}"))
            mc_conf = ast.literal_eval(row.get("mc_config", "{}"))
            match_id = row.get("Configuration_ID")
        except (ValueError, SyntaxError) as e:
            continue

        if not mc_conf:
            continue

        row_result = {"id": match_id}

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



# match_results = pd.read_csv("./ImprovedDataToMCMatches/match_results/match_results_0.5.csv")
# match_results_analyer(match_results)
