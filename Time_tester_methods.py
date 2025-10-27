import time
import os
import math
import sys
import logging
import pandas as pd
from self_repair.oversampling_methods.oversamplingMethods import RandomOversampling, LimeBasedOversampling, KDEOversampling, PlugInvalid
from self_repair.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_benchmark_all_methods(n_runs: int = 10, n_samples: int = 30):
    p = Pipeline(
        training_dataset_path="data/dataset1000.csv",
        test_data_path="data/dataset1000.csv",
        points_regressor=100,
        n_data_to_verify=10
    )

    first_30 = p.test_set.head(30)
    methods = [RandomOversampling, LimeBasedOversampling, KDEOversampling, PlugInvalid]
    rows = []
    logging.info(f"Benchmark start: {len(methods)} methods with {n_runs} run each")

    for cls in methods:
        logging.info(f"Method: {cls.__name__}")
        for run_id in range(1, n_runs + 1):
            t0 = time.perf_counter()
            ok = True
            err_msg = ""
            try:
                if cls == LimeBasedOversampling:
                    instance = cls(p.regressor)
                    instance.run_oversampling(df=first_30.copy(), n_samples=n_samples)
                elif cls == PlugInvalid:
                    instance = cls()
                    instance.run_oversampling(df=first_30.copy())
                else:
                    instance = cls()
                    instance.run_oversampling(df=first_30.copy(), n_samples=n_samples)
            except Exception as e:
                ok = False
                err_msg = repr(e)
                logging.exception(f"Error with run id: {run_id} for {cls.__name__}")
            t1 = time.perf_counter()

            rows.append({
                "Run": run_id,
                "Method": cls.__name__,
                "ExecutionTime": (t1 - t0) if ok else math.nan,
                "OK": ok,
                "Error": err_msg
            })

    df = pd.DataFrame(rows)
    os.makedirs("output/time_data_methods", exist_ok=True)
    df.to_csv("output/time_data_methods/oversampling_times_repeated.csv", index=False)
    print("Benchmark data saved to output/time_data_methods/oversampling_times_repeated.csv")

if __name__ == "__main__":
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run_benchmark_all_methods(n_runs=n_runs, n_samples=n_samples)