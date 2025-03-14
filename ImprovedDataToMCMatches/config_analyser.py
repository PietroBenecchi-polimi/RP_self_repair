"""
This script compares two datasets (optimized_set and dataset_A) to find similar configurations within a user-defined proportional epsilon range.

Usage: Input the two datasets and a given epsilon (between 0 and 1), and the function will provide an array of both optimized configuration and MC configuration, along with whether they match or not.
"""
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import ast
import math
def find_similar_configs(df_B, df_A, epsilon_percentage, original_df=None):
    """
    Find similar configurations between two datasets within a proportional epsilon range.

    Args:
        df_B (pd.DataFrame): DataFrame of the query dataset (to retain column names).
        df_A (pd.DataFrame): DataFrame of the reference dataset (to retain column names).
        epsilon_percentage (float): Proportional epsilon (e.g., 0.1 for 10%).
        original_df (pd.DataFrame, optional): Original reference DataFrame if different from df_A. Defaults to None.

    Returns:
        list: A list of dictionaries containing the query configuration, matched configuration (with column names), and match status.
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

def match_results_analyer(match_results, num_rows=10):
    output_csv_path = "./ImprovedDataToMCMatches/match_results/opt_mc_difference.csv"
    # Convert DataFrame to a list of dictionaries
    match_results = match_results.to_dict(orient='records')
    results = []

    for row in match_results[:num_rows]:
        try:
            # Parse configurations
            opt_conf = ast.literal_eval(row.get("opt_config", "{}"))
            mc_conf = ast.literal_eval(row.get("mc_config", "{}"))
            match_id = row.get("Configuration_ID")
        except (ValueError, SyntaxError) as e:
            continue

        if not mc_conf:
            continue

        # Initialize a dictionary for the current row's results
        row_result = {"id": match_id}

        # Calculate differences for each factor key
        for factor_key in mc_conf.keys():
            if factor_key in opt_conf:
                difference = abs(mc_conf[factor_key] - opt_conf[factor_key])
                row_result[f"diff_{factor_key}"] = difference
            else:
                row_result[f"diff_{factor_key}"] = None  # Handle missing keys

        # Add the row result to the results list
        results.append(row_result)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
match_results = pd.read_csv("./ImprovedDataToMCMatches/match_results/match_results_0.05.csv")
match_results_analyer(match_results)
