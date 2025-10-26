import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from self_repair.pipeline import Pipeline
from self_repair import mc_opt_interface as mc

def time_prediction_model(use_mc_results=False):
    # === Percorsi di salvataggio ===
    save_dir = "visualization"
    os.makedirs(save_dir, exist_ok=True)
    save_img_path = os.path.join(save_dir, "time_prediction_model.png")
    save_csv_path = os.path.join(save_dir, "time_prediction_model_1.csv")

    # --- Setup pipeline ---
    p = Pipeline(
        training_dataset_path="data/dataset1000.csv",
        test_data_path="data/initial_configurations_to_improve.csv",
        points_regressor=100,
        n_data_to_verify=30
    )

    mc_opt = mc.ModelCheckerInterface()

    # Rimuovo la colonna target se presente
    X = p.test_set.drop(columns=['SCS'], errors='ignore')

    prediction_times = []
    predictions = []

    n = 2
    for i in range(n):
        start = time.perf_counter()

        pred_df = mc_opt.mc_results_from_configs(X.iloc[[i]])
        pred = pred_df.iloc[0]  # estrae la prima (e unica) riga

        end = time.perf_counter()
        elapsed = end - start

        prediction_times.append(elapsed)
        predictions.append(pred)

        print(f"🕒 Prediction {i+1}/{n}: {elapsed:.6f} s")

    # --- Statistiche ---
    mean_time = float(np.mean(prediction_times))
    std_time = float(np.std(prediction_times))

    print(f"\n📊 Average prediction time: {mean_time:.6f} s")
    print(f"📉 Standard deviation:      {std_time:.6f} s")

    # --- Salva CSV ---
    df_times = pd.DataFrame({
        "Index": range(1, n + 1),
        "PredictionTime_s": prediction_times
    })
    df_times.loc[len(df_times.index)] = ["mean", mean_time]
    df_times.loc[len(df_times.index)] = ["std", std_time]
    df_times.to_csv(save_csv_path, index=False)

    print(f"✅ Saved prediction times to: {save_csv_path}")

    # --- Boxplot ---
    plt.figure(figsize=(6, 4))
    plt.boxplot(prediction_times, vert=True, patch_artist=True)
    plt.title("Prediction Time per Sample")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_img_path, dpi=300)
    plt.close()

    print(f"✅ Saved boxplot to: {save_img_path}")

    return predictions, prediction_times, mean_time, std_time


if __name__ == "__main__":
    time_prediction_model()
