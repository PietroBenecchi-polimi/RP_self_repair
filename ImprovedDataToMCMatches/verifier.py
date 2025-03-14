import pandas as pd
import math
from config_analyser import find_similar_configs

# Load datasets
optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv")  # Query dataset
parsed_mc_set = pd.read_csv("./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv")  # Reference dataset
mc_set = pd.read_csv("./datasets/initial_configurations_to_improve.csv")

# Find similar configurations
results = find_similar_configs(optimized_set, parsed_mc_set, 0.5, mc_set)

validity_array = []
FTG_threshold = 0.01
def validate_scs(opt_SCS, mc_SCS):
    """Validate the SCS parameter."""
    if opt_SCS > 0.9:
        return bool(mc_SCS)
    else:
        return not bool(mc_SCS)

def validate_ftg(mc_ftg, opt_ftg, threshold=0.1):
    """Validate the FTG parameter."""
    if mc_ftg == 0 and opt_ftg == 0:
        return True
    elif mc_ftg == 0 or opt_ftg == 0:
        return abs(mc_ftg - opt_ftg) <= threshold
    else:
        return abs(mc_ftg - opt_ftg) / max(abs(mc_ftg), abs(opt_ftg)) <= threshold

for result in results:
    if result["has_match"]:
        opt_SCS = result["opt_config"]["SCS"]
        mc_SCS = result["mc_config"]["SCS"]
        validity = validate_scs(opt_SCS, mc_SCS)

        # Uncomment to include FTG validation
        mc_ftg = result["mc_config"]["FTG_HUM_1"]
        opt_ftg = result["opt_config"]["FTG"]
        validity = validity and validate_ftg(mc_ftg, opt_ftg, FTG_threshold)
        validity_array.append(validity)

invalid_count = sum(not v for v in validity_array)
print(f"Over {len(validity_array)} comparisons, {invalid_count} were invalid, total success percentage: {round(1 - (invalid_count / len(validity_array)), 2)}%")