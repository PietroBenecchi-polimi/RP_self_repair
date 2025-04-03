import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("utils"))
from rp_logger import logger

def validate_scs(opt_SCS, mc_SCS, mc_ub = 1, mc_lb = 0):
        if opt_SCS > 0.9:
            return bool(mc_SCS) and mc_SCS <= mc_ub
        else:
            return not bool(mc_SCS) and mc_SCS >= mc_lb

def validate_ftg(mc_ftg, opt_ftg, threshold=FTG_threshold):
    if mc_ftg == 0 and opt_ftg == 0:
        return True
    else:
        return abs(mc_ftg - opt_ftg) <= threshold

def validate_configurations(opt_results, ground_truth, FTG_threshold=0.005):
    validity_array = []
    
    invalid_results = []
    # Iterate through the results and validate each configuration
    for i in range(len(opt_results)):
        opt_SCS = opt_results.iloc[i]['SCS']
        mc_SCS = ground_truth.iloc[i]['SCS']
        mc_SCS_ub = ground_truth.iloc[i]['PRSCS_UB']
        mc_SCS_lb = ground_truth.iloc[i]['PRSCS_LB']
        if not pd.isna(mc_SCS_ub) and not pd.isna(mc_SCS_lb):
            logger.debug("UB and LB validity will be assesed for SCS metric")
            validity = validate_scs(opt_SCS, mc_SCS, mc_SCS_ub, mc_SCS_lb)
        else:
            logger.warning("Only binary validity will be assesed for SCS metric")
            validity = validate_scs(opt_SCS, mc_SCS)
        validity_array.append(validity)

       # Validate FTG parameter
        mc_ftg = ground_truth["FTG"]
        opt_ftg = opt_results.iloc[i]['FTG']
        validity = validity and validate_ftg(mc_ftg, opt_ftg, FTG_threshold)

        if(not validity):
            invalid_results.append(opt_results.iloc[i])

    # Calculate validation metrics
    invalid_count = sum(not v for v in validity_array)

    if(len(validity_array) == 0):
        success_percentage = 0
    else:
        success_percentage = round(1 - (invalid_count / len(validity_array)), 2)

    invalid_results_df = pd.DataFrame(invalid_results)
    
    return invalid_results_df, success_percentage