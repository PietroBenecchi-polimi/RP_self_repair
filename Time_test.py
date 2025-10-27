import os
import sys
import time
import numpy as np
import pandas as pd
from self_repair.pipeline import Pipeline
from self_repair import mc_opt_interface as mc

def time_prediction_model(n: int = 2):
    """
    Measures prediction times of the regressor for the first `n` samples
    and saves results to a CSV file.

    Args:
        n (int): Number of test samples to evaluate.
    """
    # === Output setup ===
    save_dir = "output/time_data/"
    os.makedirs(save_dir, exist_ok=True)
    save_csv_path = os.path.join(save_dir, "time_prediction_model.csv")

    # --- Setup pipeline ---
    p = Pipeline(
        training_dataset_path="data/dataset1000.csv",
        test_data_path="data/initial_configurations_to_improve.csv",
        points_regressor=100,
        n_data_to_verify=30
    )

    mc_opt = mc.RegressorInterface(p.ground_truth_regressor)

    # --- Prepare data ---
    X = p.test_set.drop(columns=['SCS'], errors='ignore')

    prediction_times = []
    predictions = []

    for i in range(n):
        start = time.perf_counter()
        pred_df = mc_opt.mc_results_from_configs(X.iloc[[i]])
        pred = pred_df.iloc[0]
        end = time.perf_counter()
        elapsed = end - start
        prediction_times.append(elapsed)
        predictions.append(pred)

    # --- Statistics ---
    mean_time = float(np.mean(prediction_times))
    std_time = float(np.std(prediction_times))

    # --- Save results ---
    df_times = pd.DataFrame({
        "Index": range(1, n + 1),
        "PredictionTime_s": prediction_times
    })
    df_times.loc[len(df_times.index)] = ["mean", mean_time]
    df_times.loc[len(df_times.index)] = ["std", std_time]
    df_times.to_csv(save_csv_path, index=False)

    return predictions, prediction_times, mean_time, std_time


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    time_prediction_model(n)