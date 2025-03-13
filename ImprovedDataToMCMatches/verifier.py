import pandas as pd
import math
from config_analyser import find_similar_configs

# Load datasets
optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv")  # Query dataset
parsed_mc_set = pd.read_csv("./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv")  # Reference dataset
mc_set = pd.read_csv("./datasets/initial_configurations_to_improve.csv")

# Find similar configurations
results = find_similar_configs(optimized_set, parsed_mc_set, 0.05, mc_set)

FTG_threshold = 0.05
validity_array = []

for result in results:
    if result["has_match"]:
        opt_SCS = result["opt_config"]["SCS"]
        mc_SCS = result["mc_config"]["SCS"]

        # Validate SCS
        if not math.isnan(mc_SCS):
            validity = result["mc_config"]["PRSCS_UB"] <= mc_SCS
        else:
            validity = result["mc_config"]["PRSCS_LB"] >= mc_SCS

        # Validate FTG
        mc_ftg = result["mc_config"]["FTG_HUM_1"]
        opt_ftg = result["opt_config"]["FTG"]

        if mc_ftg == 0 and opt_ftg == 0:
            validity = True
        elif mc_ftg == 0 or opt_ftg == 0:
            validity = abs(mc_ftg - opt_ftg) <= FTG_threshold  # Avoid division by zero
        else:
            validity = abs(mc_ftg - opt_ftg) / max(abs(mc_ftg), abs(opt_ftg)) <= FTG_threshold

        validity_array.append(validity)

# Summary
invalid_count = sum(not v for v in validity_array)
print(f"Over {len(validity_array)} comparisons, {invalid_count} were invalid")