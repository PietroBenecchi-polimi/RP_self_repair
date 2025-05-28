import pandas as pd
import numpy as np

def validate_scs(opt_SCS, mc_SCS):
    epsilon = np.abs(opt_SCS - mc_SCS)
    return epsilon < 0.1, epsilon

def validate_configurations(opt_results, ground_truth):
    validity_array = []
    epsilon_array = []
    invalid_results = []

    # **Iterate through results and validate configurations**
    for i in range(len(opt_results)):
        opt_SCS = opt_results.iloc[i]['SCS']
        try:
            mc_SCS = ground_truth.iloc[i]['SCS']
        except IndexError:
            continue
        # Validate SCS
        validity, epsilon = validate_scs(opt_SCS, mc_SCS)

        if not validity:
            invalid_results.append(opt_results.iloc[i])

        validity_array.append(validity)
        epsilon_array.append(epsilon)

    invalid_results_df = pd.DataFrame(invalid_results)

    return invalid_results_df, epsilon_array