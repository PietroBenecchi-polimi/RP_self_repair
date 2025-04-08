import pandas as pd
import joblib
from oversampling import lime_based_resampling, random_oversampling, smote_oversampling
from validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import numpy as np

warnings.simplefilter("ignore", InconsistentVersionWarning)
from utils.rp_logger import logger

def upload_regressors(ground_truth=None, regressor=None):
    ground_truth_model = joblib.load(ground_truth)
    regressor = joblib.load(regressor)

    return regressor, ground_truth_model

def verificate(regressor=None, ground_truth=None, samples=None):
    # Upload samples
    initial_configs = pd.read_csv(samples)
    sampled_df = initial_configs.sample(n=100, random_state=42)
    sampled_df = sampled_df.drop(columns=["SCS", "FTG"])
    prediction_regressor_data = sampled_df.copy()

    #Predictions
    prediction_regressor_data['SCS'] = regressor.predict(prediction_regressor_data)
    ground_truth_data = ground_truth.predict(sampled_df)
    ground_truth_data = pd.DataFrame(ground_truth_data, columns=['SCS'])

    invalid_config, success_percantage = validate_configurations(prediction_regressor_data, ground_truth_data, FTG_threshold=0.01)
    invalid_config = pd.DataFrame(invalid_config)

    return invalid_config, success_percantage

def oversampling_methods(invalid_configs, n_samples=100, regressor=None):
    invalid_configs = invalid_configs.sample(n=n_samples) if len(invalid_configs) > n_samples else invalid_configs
    new_samples_list = []
    # Oversampling methods:
    # 1. Random oversampling
    new_samples_random = random_oversampling(df=invalid_configs)
    new_samples = {
        "method": "Random",
        "samples": new_samples_random
    }
    new_samples_list.append(new_samples)
    invalid_configs.to_csv("datasets/invalid_config.csv", index=False)
    invalid_configs = pd.read_csv("datasets/invalid_config.csv")
    # 2. SMOTE based oversampling
    new_samples_smote = smote_oversampling(df=invalid_configs)
    new_samples = {
        "method": "Smote",
        "samples": new_samples_smote
   }
    new_samples_list.append(new_samples)

    # 3. LIME based oversampling
    new_samples_lime = lime_based_resampling(df=invalid_configs, regressor=regressor)
    new_samples_lime['SCS'] = np.random.uniform(
        low=invalid_configs['SCS'].min(), 
        high=invalid_configs['SCS'].max(), 
        size=new_samples_lime.shape[0]
    )
    new_samples = {
        "method": "LIME_gaussian",
        "samples": new_samples_lime
    }
    new_samples_list.append(new_samples)

    return new_samples_list

# Possible function divison:
# 1. read files, upload regressor
# 2. Oversampling methods
# 3. Retraining function
# 4. Validation function
def main():
    # Read and upload files
    ground_truth_path = "self_repair/regressor/regressor_SCS.joblib"
    regressor_path = "self_repair/regressor/regressor_SCS_LIME_100.joblib"

    regressor, ground_truth = upload_regressors(ground_truth=ground_truth_path, regressor=regressor_path)
    invalid_config, succes_before_training = verificate(regressor=regressor, ground_truth=ground_truth, samples="datasets/last_100_rows.csv")

    # Oversampling methods: 20 features without SCS and FTG
    new_samples_list = oversampling_methods(invalid_config, n_samples=900, regressor=regressor)

    stats = []
    #Retrain regressor with oversampled data
    for new_samples in new_samples_list:
        X = new_samples.get("samples").drop(columns=["SCS"])
        y = ground_truth.predict(X)

        regressor.fit(X, y)
        # Validate results using new and unseen data in training
        _, success_percentage = verificate(regressor=regressor, ground_truth=ground_truth, samples="datasets/last_100_rows.csv")
        stats.append({
           "method": new_samples.get("method"),
           "success_percentage_improvement": (success_percentage - succes_before_training) * 100,
        })

    print(f"Success percentage before retraining: {succes_before_training}")
    for stat in stats:
        msg = f"{stat['method']}: {stat['success_percentage_improvement']}"
        logger.debug(msg)
    best_method = max(stats, key=lambda x: x['success_percentage_improvement'])
    logger.debug(f"Best resampling method is: {best_method['method']} YUPPY!")

if __name__ == "__main__":
    main()