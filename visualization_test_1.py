from utils.saving_data import load_existing_results
from visualization.visualization import process_oversampling_results

def load_and_print_results(save_path: str):
    results = load_existing_results(save_path)
    
    results = process_oversampling_results(results, "first_verification")

if __name__ == "__main__":
    save_path = "tester_results/data/data_test_1_full/oversampling_results_first_verification.pkl"
    load_and_print_results(save_path)
    
