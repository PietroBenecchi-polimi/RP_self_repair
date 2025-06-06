
from utils.saving_data import load_existing_results

if __name__ == "__main__":
   results = load_existing_results("tester_results/data/data_test_1/oversampling_results_first_verification.pkl")
   
   for i, stat in enumerate(results):
        print(stat)
        print()
