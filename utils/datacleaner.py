import pandas as pd
import os
import sys
from utils.rp_logger import logger

# Transformation rules
transform_rules = {
    'HUM_1_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_1_AGE': {'y': 0, 'e': 1},
    'HUM_1_STA': {'s': 1, 'h': 0, 'u': 2},
    'HUM_2_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_2_AGE': {'y': 0, 'e': 1},
    'HUM_2_STA': {'s': 1, 'h': 0, 'u': 2},
}

# Reverse rules
reverse_rules = {
    col: {v: k for k, v in mapping.items()} for col, mapping in transform_rules.items()
}

def get_transformation_rules():
    return transform_rules

def categorical_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for column, mapping in transform_rules.items():
        if column in df_copy.columns:
            df_copy[column] = df_copy[column].map(mapping)
    return df_copy

def numeric_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for column, mapping in reverse_rules.items():
        if column in df_copy.columns:
            df_copy[column] = df_copy[column].map(mapping)
    return df_copy

def main(input_csv_path: str, output_filename: str):
    logger.info(f"Loading input data from: {input_csv_path}")

    # Load data
    df_input = pd.read_csv(input_csv_path)

    # Transform
    df_transformed = categorical_to_numeric(df_input)

    # Save to specified output file
    output_dir = './datasets'
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    df_transformed.to_csv(output_path, index=False)

    logger.info(f"Transformation complete. Output saved to: {output_path}")

def prepare_dataset(data_path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    
    data = data.drop(["PRSCS_LB","PRSCS_UB","FTG_HUM_1_LB","FTG_HUM_1_UB",
                     "FTG_HUM_1","FTG_HUM_2_LB","FTG_HUM_2_UB","FTG_HUM_2"], axis =1)
    
    data = categorical_to_numeric(data)
    
    return data