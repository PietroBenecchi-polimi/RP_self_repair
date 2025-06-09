import json
from utils.datacleaner import get_transformation_rules
import numpy as np
import pandas as pd
from utils.rp_logger import logger

with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

def generate_neighbours_from_config(config: pd.DataFrame, regressor, neighbours_to_generate=20):
    neighbours = pd.DataFrame()

    transformation_rules = get_transformation_rules()

    # If target_config is a DataFrame with one row, convert to Series for easier access
    if isinstance(config, pd.DataFrame):
        if len(config) != 1:
            raise ValueError("target_config should be a DataFrame with exactly one row.")
        config = config.iloc[0]

    for factor_key in config.index:
        if factor_key in transformation_rules:
            # Categorical variable: sample from allowed values. Sure of random choice.
            values = list(transformation_rules[factor_key].values())
            neighbours[factor_key] = np.random.choice(values, neighbours_to_generate)
        elif factor_key in factors:
            factor_info = factors[factor_key]
            # Determine type and range
            col_min = factor_info.get("min", None)
            col_max = factor_info.get("max", None)
            is_int = factor_info.get("type", "float") == "int"
            # If no min/max, fallback to target value ±10%
            if col_min is None or col_max is None:
                center = config[factor_key]
                col_min = center * 0.9
                col_max = center * 1.1
            # sample around the target value, but within allowed range
            center = config[factor_key]
            spread = (col_max - col_min) * 0.1
            low = max(col_min, center - spread)
            high = min(col_max, center + spread)
            if low >= high:
                # fallback: it happens for PSC_TAU, strange
                neighbours[factor_key] = np.repeat(center, neighbours_to_generate)
            elif is_int:
                neighbours[factor_key] = np.random.randint(int(np.ceil(low)), int(np.floor(high)) + 1, neighbours_to_generate)
            else:
                neighbours[factor_key] = np.random.uniform(low, high, neighbours_to_generate)
        else:
            # Fallback: just repeat the target value
            neighbours[factor_key] = np.repeat(config[factor_key], neighbours_to_generate)

    neighbours['SCS'] = 0

    return neighbours