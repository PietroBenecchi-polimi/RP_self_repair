import pandas as pd
from config_analyser import find_similar_configs, match_results_analyer, validate_configurations

# Load datasets
optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv")  # Query dataset
parsed_mc_set = pd.read_csv("./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv")  # Reference dataset
mc_set = pd.read_csv("./datasets/initial_configurations_to_improve.csv")
epsilons = [0.5, 0.25, 0.15, 0.1, 0.05, 0.01]
# For each set of scenario find the possible configurations
for epsilon in epsilons:
    results = find_similar_configs(optimized_set, parsed_mc_set, epsilon, mc_set)
    num_results = len(results["data"])
    similar_configs = len([result for result in results["data"] if result["has_match"] == True])
    misses = num_results - similar_configs
    print(f"For epsilon: {epsilon}: {similar_configs} similar configurations, {misses} total misses with  {misses/num_results * 100}% of misses")
    # Creates match_error data
    match_results_analyer(results)
    #Validate configuration with different FTG_epsilon values
    for ftg_epsilon in epsilons:
        data = validate_configurations(results, ftg_epsilon)
        print(f"Analytics on match accuracy with epsilon: {epsilon} and ftg_epsilon: {ftg_epsilon}: {data['invalid_count']} wrong predictions over {data['total_comparisons']} total comparisons with  {data['success_percentage'] * 100}% of success")
