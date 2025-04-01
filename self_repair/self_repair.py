import pandas as pd
import joblib
def validate_configurations(opt_config, ground_truth, FTG_threshold=0.01):

    def validate_scs(opt_SCS, mc_SCS):
        if opt_SCS > 0.9:
            return bool(mc_SCS)
        else:
            return not bool(mc_SCS)

    def validate_ftg(mc_ftg, opt_ftg, threshold=FTG_threshold):
        if mc_ftg == 0 and opt_ftg == 0:
            return True
        else:
            return abs(mc_ftg - opt_ftg) <= threshold

    data_list = []
    # Iterate through the results and validate each configuration
    for result in opt_config:
        opt_SCS = result["SCS"]
        mc_SCS = ground_truth["SCS"]
        validity = validate_scs(opt_SCS, mc_SCS)

        # Validate FTG parameter
        mc_ftg = ground_truth["FTG"]
        opt_ftg = result["FTG"]
        validity = validity and validate_ftg(mc_ftg, opt_ftg, FTG_threshold)

        data_list.append({
                "opt_config": result["opt_config"],
                "valid": validity
            })
        validity_array.append(validity)
    # Calculate validation metrics
    invalid_count = sum(not v for v in validity_array)
    if(len(validity_array) == 0):
        success_percentage = 0
    else:
        success_percentage = round(1 - (invalid_count / len(validity_array)), 2)

    return {
        "total_data": data_list,
        "total_comparisons": len(validity_array),
        "invalid_count": invalid_count,
        "success_percentage": success_percentage
    }

initial_configs = pd.read_csv('/datasets/transformed_dataset.csv')
initial_configs = initial_configs.drop(columns=["SCS", "FTG"])
print(initial_configs)
regressor = joblib.load('regressor_SCS_LIME_100.joblib')
ground_truth = joblib.load('regressor_SCS.joblib')
optimizeds = regressor.predict(initial_configs)
print(len(predictions))

