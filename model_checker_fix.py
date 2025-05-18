import model_checker.hri_designtime.src.hmt_factors as hmtf
import pandas as pd
import self_repair.oversampling as sample
import utils.datacleaner as dc
import joblib


def check_oversampling_methods():
    data_path = "data/dataset1000.csv"
    # drop just some useless columns
    original_dataset = dc.load_dataset_for_regressor(data_path).sample(10, random_state=128).reset_index(drop=True)

    regressor = joblib.load("self_repair/regressors/regressor_SCS.joblib")

    # method 1: LIME
    dataset =  sample.lime_based_resampling(original_dataset, regressor, n_samples=100)
    dataset = dc.fromOptimizerToMC(dataset)
    #dataset = hmtf.run_mc_simulations(dataset)

    # method 2: random
    dataset =  sample.random_oversampling(original_dataset, n_samples=100)
    dataset = dc.fromOptimizerToMC(dataset)
    # dataset = hmtf.run_mc_simulations(dataset)

    # method 3: kde
    dataset = sample.kde_based_resampling(original_dataset, n_samples=100)
    dataset = dc.fromOptimizerToMC(dataset)
    dataset = hmtf.run_mc_simulations(dataset)

    # method 4: smote
    dataset = sample.smote_oversampling(original_dataset)
    dataset = dc.fromOptimizerToMC(dataset)
    #dataset = hmtf.run_mc_simulations(dataset)


if __name__ == "__main__":
    check_oversampling_methods()

