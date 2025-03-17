import pandas as pd
from config_analyser import find_similar_configs, match_results_analyer, validate_configurations

# Load datasets
optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv")  # Query dataset
parsed_mc_set = pd.read_csv("./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv")  # Reference dataset
mc_set = pd.read_csv("./datasets/initial_configurations_to_improve.csv")
epsilons = [0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]

# Initialize a list to store results
results_data = []

# For each epsilon, find similar configurations and validate
for epsilon in epsilons:
    # Find similar configurations
    results = find_similar_configs(optimized_set, parsed_mc_set, epsilon)
    num_results = len(results["data"])
    similar_configs = len([result for result in results["data"] if result["has_match"] == True])
    misses = num_results - similar_configs
    miss_percentage = misses / num_results * 100

    # Create match_error data
    match_results_analyer(results)

    results_data = []
    # Validate configurations with different FTG epsilon values
    for ftg_epsilon in epsilons:
        data = validate_configurations(results, ftg_epsilon)
        results_data.append({
            "FTG Epsilon": ftg_epsilon,
            "Similar Configurations": similar_configs,
            "Total Misses": misses,
            "Miss Percentage": miss_percentage,
            "Wrong Predictions": data['invalid_count'],
            "Total Comparisons": data['total_comparisons'],
            "Success Percentage": data['success_percentage'] * 100
        })
    results_df = pd.DataFrame(results_data)
    # Save the results to a CSV file
    results_df.to_csv(f"./ImprovedDataToMCMatches/verifier_results/analysis_results_{epsilon}.csv")
    print(f"Analysis of {epsilon} completed and saved as analysis_results_{epsilon}.csv")

print("Analysis complete. Results saved to the verifier_results folder.")