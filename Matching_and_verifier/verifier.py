import pandas as pd
import matplotlib.pyplot as plt
from config_analyser import find_similar_configs, match_results_analyer, validate_configurations
import os

def generate_data(epsilons, ftg_epsilons):
    # Load datasets
    optimized_set = pd.read_csv("./datasets/configurations_improved_20_20.csv") 
    parsed_mc_set = pd.read_csv("./Matching_and_verifier/refinedData/transformed_dataset.csv")  

    total_data = []

    # For each epsilon, find similar configurations and validate
    for epsilon in epsilons:
        # Find similar configurations
        results = find_similar_configs(optimized_set, parsed_mc_set, epsilon)
        results_df = pd.DataFrame(results["data"])
        results_df.to_csv(f"./Matching_and_verifier/match_results/matches_{epsilon}.csv")
        num_results = len(results["data"])
        similar_configs = len([result for result in results["data"] if result["has_match"] == True])
        misses = num_results - similar_configs
        miss_percentage = misses / num_results * 100

        # Create match_error data
        match_results_analyer(results)

        results_data = []
        # Validate configurations with different FTG epsilon values
        for ftg_epsilon in ftg_epsilons:
            data = validate_configurations(results, ftg_epsilon)
            results_data.append({
                "FTG Epsilon": ftg_epsilon,
                "Similar Configurations": similar_configs,
                "Total Misses": misses,
                "Miss Percentage": miss_percentage,
                "Wrong Predictions": data['invalid_count'],
                "Total Comparisons": data['total_comparisons'],
                "Success Percentage": data['success_percentage']

            })
        total_data.append(results_data)
        results_df = pd.DataFrame(results_data)
        # Save the results to a CSV file
        results_df.to_csv(f"./Matching_and_verifier/verifier_results/analysis_results_{epsilon}.csv")
        print(f"Analysis of {epsilon} completed and saved as analysis_results_{epsilon}.csv")

    print("Analysis complete. Results saved to the verifier_results folder.")

def create_plot(epsilons, ftg_epsilons):
    folder_path = "./Matching_and_verifier/verifier_results"

    # Initialize a figure
    plt.figure(figsize=(10, 6))

    # Loop through each ftg_epsilon and plot its corresponding line
    for ftg_epsilon in ftg_epsilons:
        y_values = []
        for epsilon in epsilons:
            file_path = os.path.join(folder_path, f"analysis_results_{epsilon}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Filter results for the current ftg_epsilon
                result = df[df["FTG Epsilon"] == ftg_epsilon].iloc[0]
                y_values.append(1 - result["Success Percentage"])
            else:
                print(f"File {file_path} does not exist.")
                y_values.append(None)

        # Plot the line for the current ftg_epsilon
        plt.plot(epsilons, y_values, marker='o', linestyle='-', label=f'FTG Epsilon = {ftg_epsilon}')

    # Add labels, title, legend, and grid
    plt.xlabel('Epsilon')
    plt.ylabel('Error Percentage')
    plt.title('Error Percentage vs Epsilon for Different FTG Epsilons')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    output_file = "./Matching_and_verifier/verifier_results/error_percentage_vs_epsilon_all_ftg.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
    print(f"Plot saved to {output_file}")

def main():
    print("Choose an option:")
    print("1. Generate Data")
    print("2. Create Plot")
    choice = input("Enter your choice (1 or 2): ")

    # Allow the user to input epsilons and ftg_epsilons
    epsilons_input = input("Enter epsilon values (comma-separated, e.g., 0.2,0.185,0.17): ")
    ftg_epsilons_input = input("Enter FTG epsilon values (comma-separated, e.g., 1,0.1,0.05): ")

    # Convert inputs to lists of floats
    epsilons = [float(e) for e in epsilons_input.split(",")]
    ftg_epsilons = [float(f) for f in ftg_epsilons_input.split(",")]

    if choice == '1':
        generate_data(epsilons, ftg_epsilons)
    elif choice == '2':
        create_plot(epsilons, ftg_epsilons)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()