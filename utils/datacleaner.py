import pandas as pd

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

def prepare_dataset(data_path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    
    data = data.drop(["PRSCS_LB","PRSCS_UB","FTG_HUM_1_LB","FTG_HUM_1_UB",
                     "FTG_HUM_1","FTG_HUM_2_LB","FTG_HUM_2_UB","FTG_HUM_2"], axis =1)
    
    data = categorical_to_numeric(data)
    
    return data

def fromMCtoOptimizer(data) -> pd.DataFrame:
    data['SCS'] = data['PRSCS_LOWER_BOUND'] + data['PRSCS_UPPER_BOUND'] / 2

    # Split AGE/STAT columns back into original components
    data['HUM_1_FTG'] = data['HUM_1_FTG'].astype(str)
    data['HUM_2_FTG'] = data['HUM_2_FTG'].astype(str)

    data[['HUM_1_AGE', 'HUM_1_STA']] = data['HUM_1_FTG'].str.split('/', expand=True)
    data[['HUM_2_AGE', 'HUM_2_STA']] = data['HUM_2_FTG'].str.split('/', expand=True)
    
    data = data.rename(columns={'PROGRESS': 'PRGS'})

    # Split Position columns back into original X and Y
    data['HUM_1_POS'] = data['HUM_1_POS'].astype(str)
    data['HUM_2_POS'] = data['HUM_2_POS'].astype(str)

    data[['HUM_1_POS_X', 'HUM_1_POS_Y']] = data['HUM_1_POS'].str.split(', ', expand=True).astype(float)
    data[['HUM_2_POS_X', 'HUM_2_POS_Y']] = data['HUM_2_POS'].str.split(', ', expand=True).astype(float)

    # Drop combined columns
    data.drop(columns=['HUM_1_FTG', 'HUM_2_FTG', 'HUM_1_POS', 'HUM_2_POS'], inplace=True)

    # Remove placeholder columns
    data.drop(columns=['PRSCS_LOWER_BOUND', 'PRSCS_UPPER_BOUND', 'FTG_HUM_1', 'FTG_HUM_2'], inplace=True)

    data = categorical_to_numeric(data) 
    
    return data

    
if __name__ == "__main__":
    results = pd.read_csv("mc_results.csv")
    results = fromMCtoOptimizer(results)
    results.to_csv("mc_results_transformed.csv", index=False)
