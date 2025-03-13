"""\
This script compares two datasets (optimized_set and model_checker_set) to find similar configurations within a user-defined proportional epsilon range. The epsilon is prompted from the user, and the script investigates how the number of misses changes as epsilon varies. Using a KDTree, it calculates a proportional epsilon for each query configuration, searches for matches in the reference dataset, and records whether a match is found. Results, including match statuses, are saved to a CSV file, and statistics (total data, misses, and miss percentage) are logged for analysis. The purpose is to analyze how miss rates vary with different epsilon values.

Usage: prompt the epsilon desidered and check the number of misses that result from it
"""
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from config_analyser import find_similar_configs
# Load datasets
optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv")  # Query dataset
model_checker_set = pd.read_csv("./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv")  # Reference dataset

# Define the proportional epsilon (e.g., 10% of the norm of the query point)
epsilon_percentage = float(input("Insert an epsilon percentage to compare the two datasets with (e.g., 0.1 = 10%): "))

# Run similarity search
matches = find_similar_configs(optimized_set, model_checker_set, epsilon_percentage)

# Prepare results for CSV
results = []
for i, result in enumerate(matches):
    results.append({
        "Configuration_ID": i,  # Configuration ID (index from query dataset)
        "Has_Match": result.match  # Boolean indicating whether a match was found
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
file_name = f"match_results_{epsilon_percentage}.csv"
results_df.to_csv("./match_results/" + file_name, index=False)

# Calculate statistics
total_data = len(results)
num_misses = results_df["Has_Match"].value_counts().get(False, 0)
percentage_misses = (num_misses / total_data) * 100

# Append statistics to epsilon_data.txt
with open("epsilon_data.txt", "a") as f:
    f.write(f"Epsilon Percentage: {epsilon_percentage}\n")
    f.write(f"Total Data Analyzed: {total_data}\n")
    f.write(f"Number of Misses: {num_misses}\n")
    f.write(f"Percentage of Misses: {percentage_misses:.2f}%\n")
    f.write("-" * 40 + "\n")  # Separator for readability

# Print results to console
print("Results saved to " + file_name)
print(f"Total number of data analyzed: {total_data}")
print(f"Number of misses: {num_misses}")
print(f"Percentage of misses: {percentage_misses:.2f}%")