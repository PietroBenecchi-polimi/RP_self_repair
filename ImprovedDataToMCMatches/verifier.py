from config_analyser import find_similar_configs
import pandas as pd
optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv")  # Query dataset
parsed_mc_set = pd.read_csv("./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv")  # Reference dataset
mc_set = pd.read_csv("./datasets/initial_configurations_to_improve.csv")
results = find_similar_configs(optimized_set, parsed_mc_set, 0.05, mc_set)
SCS_threashold = 0.05
validity_array = []
for result in results:
    if(result["has_match"]):
        opt_SCS = result["opt_config"]["SCS"]
        mc_SCS = result["mc_config"]["SCS"]
        if(mc_SCS):
            valid_SCS = result["mc_config"]["PRSCS_UB"] <= mc_SCS
        else:
            valid_SCS = result["mc_config"]["PRSCS_LB"] >= mc_SCS
        
        validity_array.append(valid_SCS)
print(f"Over {len(validity_array)} there were {validity_array.count(False)} invalid values")

