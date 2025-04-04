import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from LIME import explain_prediction_with_lime 
import smogn

# Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise 
#https://github.com/nickkunz/smogn?tab=readme-ov-file
def smote_oversampling(df):
    df_resampled = smogn.smoter(data=df, y='SCS')
    df_resampled = pd.DataFrame(df_resampled)
    return df_resampled

def random_oversampling(df, n_samples):
    synthetic_data = {}
        
    for column in df.columns:
        col_min, col_max = df[column].min(), df[column].max()
        synthetic_data[column] = np.random.uniform(col_min, col_max, n_samples)
    
    return pd.DataFrame(synthetic_data)

def lime_based_resampling(df, regressor):
    """
    Resampling based on LIME explanations.
    """    
    df = df.drop(columns=["SCS"])
    # Extract feature importances from LIME explanations
    new_samples = []

    explanations = explain_prediction_with_lime(df, regressor, num_features=20)

    for index in range(df.shape[0]):
        new_sample = df.iloc[index].copy()
        print(new_sample)
        # Using index since we have a dataseries, not a dataframe
        for feature in new_sample.keys():
            # Mean in original sample, while variance is the feature importance from LIME
            mean = new_sample[feature]
            variance = explanations.iloc[index][feature]

            # minimum variance
            variance = max(variance, 0.01)
            
            # new distribution
            new_value = np.random.normal(mean, np.sqrt(variance))
        
            new_sample[feature] = new_value
        
        new_samples.append(new_sample)
    
    resampled_df = pd.DataFrame(new_samples)

    return resampled_df

def main():
    csv_path = "Matching_and_verifier\invalid_configs\invalid_config_0.2_0.005.csv"
    model_path = "regressor_SCS.joblib"

    # get explanations from LIME
    explanations = explain_prediction_with_lime(csv_path, model_path, num_features=20)
    explanations_df = pd.DataFrame(explanations)
    X = pd.read_csv(csv_path)

    # new samples based on LIME explanations
    resampled_X = lime_based_resampling(X, explanations_df, X.shape[0], scale_factor=0.1)
    resampled_X_smogn = smote_oversampling(X)
    random_oversampling_X = random_oversampling(X, X.shape[0])

    output_path = "lime_resampled_data.csv"

    resampled_X.to_csv(output_path, index=False)
    resampled_X_smogn.to_csv("smote_resampled_data.csv", index=False)
    random_oversampling_X.to_csv("random_oversampling.csv", index=False)
    
if __name__ == "__main__":
    main()
