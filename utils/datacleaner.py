import pandas as pd
from typing import List, Dict
from self_repair.stats import Stat

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

def load_dataset_for_regressor(data_path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    
    data = data.drop(["PRSCS_LB","PRSCS_UB","FTG_HUM_1_LB","FTG_HUM_1_UB",
                     "FTG_HUM_1","FTG_HUM_2_LB","FTG_HUM_2_UB","FTG_HUM_2"], axis =1)
    
    data = categorical_to_numeric(data)
    
    return data

def fromOptimizerToMC(data):
    # Convert categorical columns to numeric
    data = numeric_to_categorical(data)
    # Combine AGE and STAT into a new column
    data['AGE/STAT 1'] = data['HUM_1_AGE'].astype(str) + '/' + data['HUM_1_STA'].astype(str)
    data['AGE/STAT 2'] = data['HUM_2_AGE'].astype(str) + '/' + data['HUM_2_STA'].astype(str)

    data['Position 1'] = data['HUM_1_POS_X'].astype(str) + ', ' + data['HUM_1_POS_Y'].astype(str)
    data['Position 2'] = data['HUM_2_POS_X'].astype(str) + ', ' + data['HUM_2_POS_Y'].astype(str) 

    data.drop(columns=['HUM_1_POS_X', 'HUM_1_POS_Y', 'HUM_2_POS_X', 'HUM_2_POS_Y', 'HUM_1_AGE', 'HUM_2_AGE',
                      'HUM_1_STA', 'HUM_2_STA'], inplace=True) 
                      
    # Extract new columns
    age_stat_1 = data.pop('AGE/STAT 1')
    age_stat_2 = data.pop('AGE/STAT 2')
    vel_1 = data.pop('Position 1')
    vel_2 = data.pop('Position 2')

    # Desidered position
    data.insert(9, 'AGE/STAT 1', age_stat_1)
    data.insert(11, 'AGE/STAT 2', age_stat_2)
    data.insert(12, 'Position 1', vel_1)
    data.insert(13, 'Position 2', vel_2)
    
    data['PRSCS_LOWER_BOUND'] = 0
    data['PRSCS_UPPER_BOUND'] = 0
    data['FTG_HUM_1'] = 0
    data['FTG_HUM_2'] = 0

    return data

def fromMCtoOptimizer(data) -> pd.DataFrame:
    mask = ~((data['PRSCS_LOWER_BOUND'] == 0) & (data['PRSCS_UPPER_BOUND'] == 0) & (data['FTG_HUM_1'] == 0) & (data['FTG_HUM_2'] == 0))
    data = data[mask]

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

    # drop combined columns
    data.drop(columns=['HUM_1_FTG', 'HUM_2_FTG', 'HUM_1_POS', 'HUM_2_POS'], inplace=True)

    # Remove columns scs and ftg(should be changed if test on fatigue)
    data.drop(columns=['PRSCS_LOWER_BOUND', 'PRSCS_UPPER_BOUND', 'FTG_HUM_1', 'FTG_HUM_2'], inplace=True)
    
    # Extract new columns
    age_1 = data.pop('HUM_1_AGE')
    age_2 = data.pop('HUM_2_AGE')
    stat_1 = data.pop('HUM_1_STA')
    stat_2 = data.pop('HUM_2_STA')
    vel_1 = data.pop('ROB_1_VEL')
    chg_1 = data.pop('ROB_1_CHG')

    # Desidered position
    data.insert(9, 'HUM_1_AGE', age_1)
    data.insert(11, 'HUM_2_AGE', age_2)
    data.insert(10, 'HUM_1_STA', stat_1)
    data.insert(13, 'HUM_2_STA', stat_2)
    data.insert(19, 'ROB_1_VEL', vel_1)
    data.insert(20, 'ROB_1_CHG', chg_1)
    
    data = categorical_to_numeric(data) 
    
    return data

def castIntegerFeatures(data):
    for feature in data.columns:
        if feature in ["PSCS__TAU", "PRGS"]:
            data[feature] = data[feature].astype(int)
    return data

def process_results(stats_per_points: List[Dict]) -> pd.DataFrame:
    """Flatten the list of stats into a DataFrame suitable for boxplotting."""
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']
        stats: list[Stat] = experiment['stats']

        for stat in stats:
            method = stat.get_method_name()
            epsilon_points = stat.get_epsilon_points()
            if(isinstance(epsilon_points, list)):
                for eps in stat.get_epsilon_points():
                    data.append({
                        'Method': method,
                        'Regressor Points': r_points,
                        'Resampling Points': s_points,
                        'Epsilons': eps
                    })
            elif(isinstance(epsilon_points, Dict)):
                for eps in epsilon_points.values():
                    data.append({
                        'Method': method,
                        'Regressor Points': r_points,
                        'Resampling Points': s_points,
                        'Epsilons': eps
                    })
    return pd.DataFrame(data)
