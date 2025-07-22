import pandas as pd
import json
import utils.datacleaner as ut
import numpy as np
import self_repair.pipeline as pipeline
import visualization.visualization as vis
import utils.datacleaner as dc
from self_repair.mc_opt_interface import ModelCheckerInterface

def main():
    p = pipeline.Pipeline(
        training_dataset_path="data/dataset1000.csv",
        test_data_path="data/initial_configurations_to_improve.csv",
        points_regressor=1000,
        n_data_to_verify=100
    )

    mc_opt_interface = ModelCheckerInterface()
    results = mc_opt_interface.mc_results_from_configs(p.test_set)
    
    results.to_csv("results.csv")

if __name__ == "__main__":
    main()