import pandas as pd
import numpy as np
from self_repair.LIME import explain_prediction_with_lime 
import smogn
import json
from utils.datacleaner import get_transformation_rules
import utils.datacleaner as dc
from sklearn.neighbors import KernelDensity

with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

# Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise 
#https://github.com/nickkunz/smogn?tab=readme-ov-file
def smote_oversampling(df: pd.DataFrame):
    df_resampled = smogn.smoter(
        data=df.reset_index(drop=True),
        y='SCS',
        k=6,
        rel_coef=0.35
    )
    
    new_data = dc.castIntegerFeatures(df_resampled)
    
    return new_data

def random_oversampling(df, n_samples = 100):
    synthetic_data = {}
    transformation_rules = get_transformation_rules()

    for factor_key in df.columns:
        if factor_key == "SCS":
            synthetic_data[factor_key] = np.random.uniform(0,1, n_samples)
        elif factor_key not in transformation_rules.keys():
            if "PRGS" == factor_key:
                synthetic_data[factor_key] = np.random.choice([0,1,2,3,4,5])
            elif factor_key == "PSCS__TAU":
                synthetic_data[factor_key] = np.random.choice(np.arange(250, 751), n_samples)
            else:
                if factor_key in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                    col_max, col_min = factors["HUM_1_POS"]["max_x"], factors["HUM_1_POS"]["min_x"]
                elif factor_key in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                    col_max, col_min = factors["HUM_1_POS"]["max_y"], factors["HUM_1_POS"]["min_y"]
                elif "max" in factors[factor_key]:
                    col_max, col_min = factors[factor_key]["max"], factors[factor_key]["min"]

                synthetic_data[factor_key] = np.random.uniform(col_min, col_max, n_samples)
        else:
            values = list(transformation_rules[factor_key].values())
            synthetic_data[factor_key] = np.random.choice(values, n_samples)

    return pd.DataFrame(synthetic_data)

def lime_based_resampling(df: pd.DataFrame, regressor, n_samples=100):
    df = df.drop(columns=["SCS"], errors='ignore').reset_index(drop=True)
    new_samples = []
    epsilon = 1e-5  # to avoid division by zero
    transformation_rules = get_transformation_rules()
    
    explanations = explain_prediction_with_lime(df, regressor, num_features=20)
    
    for _ in range(n_samples):
        index = df.sample(n=1).index[0]
        new_sample = df.loc[[index]].copy()
        
        for feature in new_sample.columns:
            mean = float(new_sample[feature].values[0])
            importance = explanations.loc[index].get(feature, 0.0)
            variance = abs(1.0 / (importance + epsilon))
            variance = min(variance, 1.0)
            
            if feature in transformation_rules:
                values = np.array(list(transformation_rules[feature].values()), dtype=float)
                probabilities = np.exp(-0.5 * ((values - mean) / variance) ** 2)
                probabilities /= probabilities.sum()
                new_value = np.random.choice(values, p=probabilities)
            else:
                if feature == "PRGS":
                    values = np.array([0, 1, 2, 3, 4, 5])
                    probabilities = np.exp(-0.5 * ((values - mean) / variance) ** 2)
                    probabilities /= probabilities.sum()
                    new_value = np.random.choice(values, p=probabilities)
                elif feature == "PSCS__TAU":
                    values = np.arange(250, 751)
                    probabilities = np.exp(-0.5 * ((values - mean) / variance) ** 2)
                    probabilities /= probabilities.sum()
                    new_value = np.random.choice(values)
                else:
                    if feature in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                        col_max, col_min = factors["HUM_1_POS"]["max_x"], factors["HUM_1_POS"]["min_x"]
                    elif feature in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                        col_max, col_min = factors["HUM_1_POS"]["max_y"], factors["HUM_1_POS"]["min_y"]
                    else:
                        try:
                            col_max, col_min = factors[feature]["max"], factors[feature]["min"]
                        except KeyError:
                            raise ValueError(f"Missing factor information for feature: {feature}")
                    
                    new_value = np.random.uniform(col_min, col_max)
            
            new_sample[feature] = new_value
        
        new_samples.append(new_sample)
    
    return pd.concat(new_samples, ignore_index=True)

def kde_based_resampling(df: pd.DataFrame, n_samples=100):
    df = df.drop(columns=["SCS"], errors='ignore')
    transformation_rules = get_transformation_rules()

    # Fit a KDE for each feature
    feature_samples = {}

    for feature in df.columns:
        if feature not in transformation_rules:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
            kde.fit(df[[feature]])
            # Sample n_samples all at once
            samples = kde.sample(n_samples).flatten()
            feature_samples[feature] = samples
        else:
            # Takes the values of each feature from the transformation rules
            values = np.array(list(transformation_rules[feature].values()), dtype=float)
            # all props are equal
            probs = np.ones(len(values)) / len(values)
            samples = np.random.choice(values, size=n_samples, p=probs)
            feature_samples[feature] = samples
    
    new_samples = pd.DataFrame(feature_samples)
    new_samples = dc.castIntegerFeatures(new_samples)

    return new_samples
