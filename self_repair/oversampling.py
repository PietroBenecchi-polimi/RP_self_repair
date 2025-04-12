import pandas as pd
import numpy as np
from self_repair.LIME import explain_prediction_with_lime 
import smogn
import json
from utils.datacleaner import get_transformation_rules
with open('datasets/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

# Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise 
#https://github.com/nickkunz/smogn?tab=readme-ov-file
def smote_oversampling(df: pd.DataFrame):
    df_resampled = smogn.smoter(
        data=df.reset_index(drop=True),
        y='SCS',
        k=6,
    )
    return pd.DataFrame(df_resampled)

def random_oversampling(df):
    synthetic_data = {}
    transformation_rules = get_transformation_rules()
    for factor_key in df.columns:
        if factor_key == "SCS":
            synthetic_data[factor_key] = np.random.uniform(0,1, len(df))
        elif factor_key not in transformation_rules.keys():
            if "PRGS" == factor_key:
                synthetic_data[factor_key] = np.random.choice([0,1,2,3,4,5])
                continue
            elif factor_key in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                col_min, col_max = factors["HUM_1_POS"]["max_x"], factors["HUM_1_POS"]["min_x"]
            elif factor_key in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                col_min, col_max = factors["HUM_1_POS"]["max_y"], factors["HUM_1_POS"]["min_y"]
            elif "max" in factors[factor_key]:
                col_min, col_max = factors[factor_key]["max"], factors[factor_key]["min"]
            synthetic_data[factor_key] = np.random.uniform(col_min, col_max, len(df))
        else:
            values = list(transformation_rules[factor_key].values())
            synthetic_data[factor_key] = np.random.choice(values, len(df))

    return pd.DataFrame(synthetic_data)

def lime_based_resampling(df, regressor):
    df = df.drop(columns=["SCS"])
    new_samples = []
    epsilon = 1e-5  # to avoid division by zero

    explanations = explain_prediction_with_lime(df, regressor, num_features=20)

    for index in range(df.shape[0]):
        new_sample = df.iloc[index].copy()
        for feature in new_sample.keys():
            mean = new_sample[feature]
            importance = explanations.iloc[index].get(feature, 0.0)
            variance = abs(1.0 / (importance + epsilon))
            variance = min(variance, 1.0)

            new_value = np.random.normal(mean, np.sqrt(variance))
            new_sample[feature] = new_value
        
        new_samples.append(new_sample)
    
    return pd.DataFrame(new_samples)
