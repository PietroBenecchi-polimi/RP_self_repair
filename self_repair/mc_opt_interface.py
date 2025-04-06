import pandas as pd
from rp_logger import logger
def mc_results_from_configs(new_configs:pd.DataFrame, cache_file_name:str) -> pd.DataFrame:
    # Try fecthing in the cache
    try:
        result = pd.read_csv(f"/self_repair/cache/mc/{cache_file_name}.csv")
    except FileNotFoundError:
        logger.warning(f"Cache miss on {cache_file_name}, MC will be used")

    return pd.DataFrame
def opt_optimization(new_configs:pd.DataFrame, cache_file_name:str) -> pd.DataFrame:
    # Try fecthing in the cache
    try:
        result = pd.read_csv(f"/self_repair/cache/opt/{cache_file_name}.csv")
    except FileNotFoundError:
        logger.warning(f"Cache miss on {cache_file_name}, MC will be used")
    return pd.DataFrame