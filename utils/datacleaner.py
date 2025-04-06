import pandas as pd
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Transformation rules
transform_rules = {
    'HUM_1_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_1_AGE': {'y': 0.0, 'e': 1.0},
    'HUM_1_STA': {'s': 1.0, 'h': 0.0, 'u': 2.0},
    'HUM_2_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_2_AGE': {'y': 0.0, 'e': 1.0},
    'HUM_2_STA': {'s': 1.0, 'h': 0.0, 'u': 2.0},
}

# Reverse rules
reverse_rules = {
    col: {v: k for k, v in mapping.items()} for col, mapping in transform_rules.items()
}

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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transform_data.py <input_csv_path> <output_filename>")
    else:
        main(sys.argv[1], sys.argv[2])