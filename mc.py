import pandas as pd
from utils.datacleaner import load_dataset_for_regressor, fromOptimizerToMC, fromMCtoOptimizer

def main():
    data = load_dataset_for_regressor('data/initial_configurations_to_improve.csv')
    data = fromOptimizerToMC(data)
    print(data.head())
    data = fromMCtoOptimizer(data)
    print(data.head())

if __name__ == "__main__":
    main()