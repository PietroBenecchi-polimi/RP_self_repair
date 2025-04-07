import pandas as pd
import joblib
from oversampling import lime_based_resampling, random_oversampling, smote_oversampling
from validation import validate_configurations
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import numpy as np
from sklearn.metrics import log_loss

warnings.simplefilter("ignore", InconsistentVersionWarning)
import logging

# Configure logger
logger = logging.getLogger("self_repair")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def upload_samples_and_regressor(ground_truth=None, regressor=None, samples=None, SCS_threshold=0.01):
    # Upload models
    ground_truth_model = joblib.load(ground_truth)
    regressor = joblib.load(regressor)

    # Upload samples
    initial_configs = pd.read_csv(samples)
    sampled_df = initial_configs.sample(n=100, random_state=42)
    sampled_df = initial_configs.drop(columns=["SCS", "FTG"])

    #Predictions
    sampled_df['SCS'] = regressor.predict(sampled_df)
    ground_truth_data = ground_truth_model.predict(sampled_df)
    ground_truth_data = [True if result > 0.5 else False for result in ground_truth_data]
    ground_truth_data = pd.DataFrame(ground_truth_data, columns=['SCS'])

    invalid_config, success_percantage = validate_configurations(sampled_df, ground_truth_data, FTG_threshold=0.01)
    invalid_config = pd.DataFrame(invalid_config)

    return invalid_config, regressor, ground_truth_model, success_percantage

def validate_configurations_after_retraining(regressor, ground_truth):
    data = pd.read_csv("datasets\configurations_improved_20_20.csv")
    new_data = data.drop(columns=["SCS", "FTG"])

    ground_truth_data = ground_truth.predict(new_data)
    ground_truth_data = [True if result > 0.5 else False for result in ground_truth_data]
    ground_truth_data = pd.DataFrame(ground_truth_data, columns=['SCS'])
    new_data['SCS'] = regressor.predict(new_data)
    
    _, success_parcantage = validate_configurations(new_data, ground_truth_data, FTG_threshold=0.01)

    #binary_cross_loss_function = log_loss(new_data['SCS'], ground_truth_data['SCS'])

    return success_parcantage, 0.0 #binary_cross_loss_function

def oversampling_methods(invalid_config, n_samples=100, regressor=None):
    new_samples_list = []
    # Oversampling methods:
    # 1. Random oversampling
    new_samples_random = random_oversampling(df=invalid_config, n_samples=n_samples)
    new_samples = {
        "method": "Random",
        "samples": new_samples_random
    }
    new_samples_list.append(new_samples)
    # 2. SMOTE based oversampling
<<<<<<< HEAD
#    new_samples_smote = smote_oversampling(df=invalid_config)
#    new_samples = {
#        "method": "Smote",
#        "samples": new_samples_smote
#   }
#    new_samples_list.append(new_samples)
=======
#     new_samples_smote = smote_oversampling(df=invalid_config)
#     new_samples = {
#         "method": "Smote",
#         "samples": new_samples_smote
#    }
#     new_samples_list.append(new_samples)
>>>>>>> 6cac96c163e9fa4f6b7c200f0fe8cd357ec74c3c

    # 3. LIME based oversampling
    new_samples_lime = lime_based_resampling(df=invalid_config, regressor=regressor)
    new_samples_lime['SCS'] = np.random.uniform(
        low=invalid_config['SCS'].min(), 
        high=invalid_config['SCS'].max(), 
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

    invalid_config, regressor, ground_truth, succes_before_training = upload_samples_and_regressor(ground_truth=ground_truth_path, regressor=regressor_path, samples='datasets/configurations_improved_20_20.csv')

    # Oversampling methods: 20 features without SCS and FTG
    new_samples_list = oversampling_methods(invalid_config, n_samples=100, regressor=regressor)

    stats = []
    #Retrain regressor with oversampled data
    for new_samples in new_samples_list:
        X = new_samples.get("samples").drop(columns=["SCS"])
        y = ground_truth.predict(X)
        regressor.fit(X, y)
        # Validate results using new and unseen data in training
        success_percentage, binary_loss = validate_configurations_after_retraining(regressor, ground_truth)
        stats.append({
           "method": new_samples.get("method"),
           "success_percentage_improvement": success_percentage - succes_before_training,
            "binary cross loss function": binary_loss,
        })

    print(f"Success percentage before retraining: {succes_before_training}")
    for stat in stats:
        msg = f"{stat['method']}: {stat['success_percentage_improvement']}"
        logger.debug(msg)
    best_method = max(stats, key=lambda x: x['success_percentage_improvement'])
    logger.debug(f"Best resampling method is: {best_method['method']} YUPPY!")

if __name__ == "__main__":
    main()