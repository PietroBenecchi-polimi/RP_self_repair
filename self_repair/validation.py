import pandas as pd
from utils.rp_logger import logger
import numpy as np
def validate_scs(opt_SCS, mc_SCS, mc_ub=1, mc_lb=0):
    epsilon = np.abs(opt_SCS - mc_SCS)
    return epsilon < 0.1
    # mc_SCS_bool = bool(mc_SCS > 0.5)
    # if opt_SCS > 0.9:
    #     return mc_SCS_bool and mc_SCS <= mc_ub
    # else:
    #     return not mc_SCS_bool and mc_SCS >= mc_lb

def validate_configurations(opt_results, ground_truth, FTG_threshold=0.005):
    validity_array = []
    invalid_results = []

    # **Pre-check if required columns are present**
    has_ub_lb = "PRSCS_UB" in ground_truth.columns and "PRSCS_LB" in ground_truth.columns

    # Log the findings
    if has_ub_lb:
        logger.info("Upper and lower bounds (PRSCS_UB, PRSCS_LB) detected for SCS validation.")
    else:
        logger.warning("Only binary validity will be assessed for SCS metric (no PRSCS_UB/PRSCS_LB found).")

    # **Iterate through results and validate configurations**
    for i in range(len(opt_results)):
        opt_SCS = opt_results.iloc[i]['SCS']
        try:
            mc_SCS = ground_truth.iloc[i]['SCS']
        except IndexError:
            continue
        # Validate SCS
        if has_ub_lb:
            mc_SCS_ub = ground_truth.iloc[i]['PRSCS_UB']
            mc_SCS_lb = ground_truth.iloc[i]['PRSCS_LB']
            validity = validate_scs(opt_SCS, mc_SCS, mc_SCS_ub, mc_SCS_lb)
        else:
            validity = validate_scs(opt_SCS, mc_SCS)

        if not validity:
            invalid_results.append(opt_results.iloc[i])

        validity_array.append(validity)

    # **Calculate validation metrics**
    invalid_count = sum(not v for v in validity_array)
    success_percentage = 0 if len(validity_array) == 0 else round(1 - (invalid_count / len(validity_array)), 2)

    invalid_results_df = pd.DataFrame(invalid_results)

    return invalid_results_df, success_percentage