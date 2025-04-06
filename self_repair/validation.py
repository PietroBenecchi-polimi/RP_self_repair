import pandas as pd
from rp_logger import logger

def validate_scs(opt_SCS, mc_SCS, mc_ub=1, mc_lb=0):
    if opt_SCS > 0.9:
        return bool(mc_SCS) and mc_SCS <= mc_ub
    else:
        return not bool(mc_SCS) and mc_SCS >= mc_lb

def validate_ftg(mc_ftg, opt_ftg, threshold):
    if mc_ftg == 0 and opt_ftg == 0:
        return True
    else:
        return abs(mc_ftg - opt_ftg) <= threshold

def validate_configurations(opt_results, ground_truth, FTG_threshold=0.005):
    validity_array = []
    invalid_results = []

    # **Pre-check if required columns are present**
    has_FTG = "FTG_HUM_1" in ground_truth.columns
    has_ub_lb = "PRSCS_UB" in ground_truth.columns and "PRSCS_LB" in ground_truth.columns

    # Log the findings
    if has_FTG:
        logger.info("FTG_HUM_1 column detected and will be assessed.")
    else:
        logger.warning("FTG_HUM_1 column is missing, it will not be assessed.")

    if has_ub_lb:
        logger.info("Upper and lower bounds (PRSCS_UB, PRSCS_LB) detected for SCS validation.")
    else:
        logger.warning("Only binary validity will be assessed for SCS metric (no PRSCS_UB/PRSCS_LB found).")

    # **Iterate through results and validate configurations**
    for i in range(len(opt_results)):
        opt_SCS = opt_results.iloc[i]['SCS']
        mc_SCS = ground_truth.iloc[i]['SCS']

        # Validate SCS
        if has_ub_lb:
            mc_SCS_ub = ground_truth.iloc[i]['PRSCS_UB']
            mc_SCS_lb = ground_truth.iloc[i]['PRSCS_LB']
            validity = validate_scs(opt_SCS, mc_SCS, mc_SCS_ub, mc_SCS_lb)
        else:
            validity = validate_scs(opt_SCS, mc_SCS)

        # Validate FTG if available
        if has_FTG:
            mc_ftg = ground_truth.iloc[i]['FTG_HUM_1']
            opt_ftg = opt_results.iloc[i]['FTG_HUM_1']
            validity = validity and validate_ftg(mc_ftg, opt_ftg, FTG_threshold)

        if not validity:
            invalid_results.append(opt_results.iloc[i])

        validity_array.append(validity)

    # **Calculate validation metrics**
    invalid_count = sum(not v for v in validity_array)
    success_percentage = 0 if len(validity_array) == 0 else round(1 - (invalid_count / len(validity_array)), 2)

    invalid_results_df = pd.DataFrame(invalid_results)

    return invalid_results_df, success_percentage