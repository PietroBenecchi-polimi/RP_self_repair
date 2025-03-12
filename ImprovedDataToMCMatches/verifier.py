import numpy as np
import pandas as pd
from scipy.spatial import KDTree
# Load datasets
optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv").to_numpy()  # Query dataset
model_checker_set = pd.read_csv("./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv").to_numpy()  # Reference dataset

# Define the proportional epsilon (e.g., 10% of the norm of the query point)
epsilon_percentage = float(input("Insert an epsilon percentage to compare the two datasets with (e.g., 0.1 = 10%): "))

# Build KDTree for the reference dataset
tree = KDTree(model_checker_set)

# Function to find nearest valid match with proportional epsilon
def find_similar_configs(dataset_B, tree, epsilon_percentage):
    results = []
    for config in dataset_B:
        # Calculate the Euclidean norm of the query point
        norm_config = np.linalg.norm(config)
        
        # Calculate proportional epsilon as a percentage of the norm
        proportional_epsilon = epsilon_percentage * norm_config
        
        # Find all points within the proportional epsilon range
        indices = tree.query_ball_point(config, r=proportional_epsilon)  # Find all within range
        
        if indices:  # If at least one valid match exists
            # Find the closest match among the candidates
            closest_idx = min(indices, key=lambda i: np.linalg.norm(model_checker_set[i] - config))
            results.append((config, model_checker_set[closest_idx], closest_idx, True))  # Store (query, closest match, index, match status)
        else:
            results.append((config, None, None, False))  # No match found
    return results

# Run similarity search
matches = find_similar_configs(optimized_set, tree, epsilon_percentage)

# Prepare results for CSV
results = []
for i, (query, match, idx, has_match) in enumerate(matches):
    results.append({
        "Configuration_ID": i,  # Configuration ID (index from query dataset)
        "Has_Match": has_match  # Boolean indicating whether a match was found
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