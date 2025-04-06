import pandas as pd
from rp_logger import logger
from NSGA_II_adapter import optimize_configurations
def mc_results_from_configs(new_configs:pd.DataFrame, cache_file_name:str) -> pd.DataFrame:
    # Try fecthing in the cache
    try:
        result = pd.read_csv(f"/self_repair/cache/mc/{cache_file_name}.csv")
    except FileNotFoundError:
        logger.warning(f"Cache miss on {cache_file_name}, MC will be used")

    return pd.DataFrame
def opt_optimization(new_configs:pd.DataFrame, scs_regressor, ftg_regressor, cache_file_name:str) -> pd.DataFrame:
    # Try fecthing in the cache
    try:
        result = pd.read_csv(f"/self_repair/cache/opt/{cache_file_name}.csv")
    except FileNotFoundError:
        logger.warning(f"Cache miss on {cache_file_name}, NSGA_II will be used")
        result = optimize_configurations(new_configs, scs_regressor, ftg_regressor)
        result.to_csv(f"/self_repair/cache/opt/{cache_file_name}")
    return result