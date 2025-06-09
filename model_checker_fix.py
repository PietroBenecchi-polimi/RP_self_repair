import pandas as pd
from self_repair.mc_opt_interface import ModelCheckerInterface
from self_repair.pipeline import Pipeline
from utils.datacleaner import fromOptimizerToMC
from utils.generateData import generate_neighbours_from_config

def main():
    data = pd.read_csv("data/initial_configurations_to_improve.csv")

    model_checker = ModelCheckerInterface(skip_cache=True)

    for i in range(len(data)):
        config = data.iloc[[i]]  # doppie parentesi per mantenere il tipo DataFrame
        configs = generate_neighbours_from_config(config)
        configs.to_csv(f"configs_{i}.csv", index=False)
        model_checker.mc_results_from_configs(configs, cache_file_name=f"config_{i}")
    
if __name__ == "__main__":
    main()