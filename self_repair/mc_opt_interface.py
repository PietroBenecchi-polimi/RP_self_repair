import pandas as pd
from utils.rp_logger import logger
from optimizer.NSGA_II_adapter import optimize_configurations
def mc_results_from_configs(new_configs:pd.DataFrame, cache_file_name:str) -> pd.DataFrame:
    # Try fecthing in the cache
    try:
        result = pd.read_csv(f"/self_repair/cache/mc/{cache_file_name}.csv")
    except FileNotFoundError:
        logger.warning(f"Cache miss on {cache_file_name}, MC will be used")
        result = pd.DataFrame(columns=pd.read_csv("datasets/initial_configurations_to_improve.csv").columns)
    return result
def mc_results_from_configs(new_configs:pd.DataFrame, scs_ground_truth_regressor) -> pd.DataFrame:
    scs_y = scs_ground_truth_regressor.predict(new_configs)
    new_configs["SCS"] = scs_y
    return new_configs

# def opt_optimization(new_configs:pd.DataFrame, scs_regressor, cache_file_name:str) -> pd.DataFrame:
#     # Try fecthing in the cache
#     try:
#         result = pd.read_csv(f"self_repair/cache/opt/{cache_file_name}.csv")
#     except FileNotFoundError:
#         logger.warning(f"Cache miss on {cache_file_name}, NSGA_II will be used")
#         result = optimize_configurations(new_configs.reset_index(), scs_regressor)
#         result.to_csv(f"self_repair/cache/opt/{cache_file_name}.csv", index=False)
#     return result.drop(columns=["FTG"])

def opt_optimization(new_configs:pd.DataFrame, scs_regressor, cache_file_name: str) -> pd.DataFrame:
    new_configs["SCS"] = scs_regressor.predict(new_configs)
    return new_configs