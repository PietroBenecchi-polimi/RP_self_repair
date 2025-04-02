import pandas as pd
import joblib
from oversampling import random_oversampling, lime_based_resampling, smote_oversampling

def validate_configurations(opt_config, ground_truth, FTG_threshold=0.01):
    validity_array = []
    def validate_scs(opt_SCS, mc_SCS):
        if opt_SCS > 0.9:
            return bool(mc_SCS)
        else:
            return not bool(mc_SCS)

#    def validate_ftg(mc_ftg, opt_ftg, threshold=FTG_threshold):
#        if mc_ftg == 0 and opt_ftg == 0:
#            return True
#        else:
#            return abs(mc_ftg - opt_ftg) <= threshold
    data_list = []
    # Iterate through the results and validate each configuration
    for i in range(len(opt_config)):
        opt_SCS = opt_config.iloc[i]['SCS']
        mc_SCS = ground_truth.iloc[i]['SCS']
        validity = validate_scs(opt_SCS, mc_SCS)

        # Validate FTG parameter
#      mc_ftg = ground_truth["FTG"] 
#      opt_ftg = opt_config.iloc[i]['FTG']
#      validity = validity and validate_ftg(mc_ftg, opt_ftg, FTG_threshold)

        if(not validity):
            data_list.append(opt_config.iloc[i])

    # Calculate validation metrics
#    invalid_count = sum(not v for v in validity_array)
#    if(len(validity_array) == 0):
#        success_percentage = 0
#    else:
#        success_percentage = round(1 - (invalid_count / len(validity_array)), 2)

    df = pd.DataFrame(data_list)
    df.to_csv("invalid_config.csv", index=False)
    print(df)
    return df


def upload_samples_and_regressor(ground_truth=None, regressor=None, samples=None, SCS_threshold=0.01):
    # Upload models
    ground_truth = joblib.load(ground_truth)
    regressor = joblib.load(regressor)

    # Upload samples
    initial_configs = pd.read_csv(samples)
    sampled_df = initial_configs.sample(n=100, random_state=42)
    sampled_df = initial_configs.drop(columns=["SCS", "FTG"])

    #Predictions
    sampled_df['SCS'] = regressor.predict(sampled_df)
    ground_truth = ground_truth.predict(sampled_df)
    ground_truth = pd.DataFrame(ground_truth, columns=['SCS'])

    invalid_config = validate_configurations(sampled_df, ground_truth, FTG_threshold=0.01)
    invalid_config = pd.DataFrame(invalid_config)

    return invalid_config, regressor

def oversampling_methods(invalid_config, n_samples=100):
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
#    new_samples_smote = smote_oversampling(df=invalid_config)
#    new_samples = {
#        "method": "Smote",
#        "samples": new_samples_smote
#   }
#    new_samples_list.append(new_samples)

    # 3. LIME based oversampling
    #TODO

    return new_samples_list

# Possible function divison:
# 1. read files, upload regressor
# 2. Oversampling methods
# 3. Retraining function
# 4. Validation function
def main():
    # Read and upload files
    ground_truth = "regressor_SCS.joblib"
    regressor_path = "regressor_SCS_LIME_100.joblib"

    invalid_config, regressor = upload_samples_and_regressor(ground_truth=ground_truth, regressor=regressor_path, samples='datasets/transformed_dataset.csv')

    # Oversampling methods
    new_samples_list = oversampling_methods(invalid_config, n_samples=100)

    #Retrain regressor with oversampled data
    for new_samples in new_samples_list:
        regressor.fit(new_samples.get("samples").drop(columns=["SCS"]), new_samples.get("samples")["SCS"])
        # Validate results using new and unseen data in training
        #validate_configurations(new_samples.get("samples"), ground_truth, FTG_threshold=0.01)

    # Restore function, something like early stopping so that I can take the best model

if __name__ == "__main__":
    main()