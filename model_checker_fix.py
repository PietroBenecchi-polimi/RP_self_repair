import pandas as pd
from self_repair.mc_opt_interface import ModelCheckerInterface
from self_repair.pipeline import Pipeline
from utils.datacleaner import fromOptimizerToMC, fromMCtoOptimizer

def main():
    p = Pipeline(training_dataset_path="data/dataset1000.csv",
                 test_data_path="data/initial_configurations_to_improve.csv",
                 points_regressor=100, n_data_to_verify=1)
    
    model_checker = ModelCheckerInterface(skip_cache=True)

    opt_configs = model_checker.opt_optimization(p.test_set, p.regressor, f"regressor_10")
    ground_truth_first_test = model_checker.mc_results_from_configs(opt_configs.drop(columns=["SCS"]), "null")
    ground_truth_first_test.to_csv("ground_truth_first_test.csv", index=False)
    invalid_configs, epsilon_array = p.validate_configurations(opt_configs, ground_truth_first_test)


if __name__ == "__main__":
    main()