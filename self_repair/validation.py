import pandas as pd

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
        validity_array.append(validity)

       # Validate FTG parameter
#      mc_ftg = ground_truth["FTG"] 
#      opt_ftg = opt_config.iloc[i]['FTG']
#      validity = validity and validate_ftg(mc_ftg, opt_ftg, FTG_threshold)

        if(not validity):
            data_list.append(opt_config.iloc[i])

    # Calculate validation metrics
    invalid_count = sum(not v for v in validity_array)

    if(len(validity_array) == 0):
        success_percentage = 0
    else:
        success_percentage = round(1 - (invalid_count / len(validity_array)), 2)

    df = pd.DataFrame(data_list)
    
    return df, success_percentage