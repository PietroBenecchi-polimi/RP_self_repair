import pandas as pd
from utils.rp_logger import logger
import model_checker.hri_designtime.src.hmt_factors as hmtf
from configurations_optimizer.NSGA_II_adapter import optimize_configurations
import os
import utils.datacleaner as dc
from abc import ABC, abstractmethod

class MC_OPT_INTERFACE(ABC):
    @abstractmethod
    def mc_results_from_configs(self, new_configs: pd.DataFrame, cache_file_name: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def opt_optimization(self, new_configs: pd.DataFrame, cache_file_name: str, skip_caching=False) -> pd.DataFrame:
        pass

class ModelCheckerInterface(MC_OPT_INTERFACE):
    def __init__(self, skip_cache: bool):
        super().__init__()
        self.skip_cache = skip_cache

    def mc_results_from_configs(self, new_configs: pd.DataFrame, cache_file_name: str) -> pd.DataFrame:
        cache_path = os.path.join("self_repair/cache/mc", f"{cache_file_name}.csv")
        
        if not self.skip_cache:
            try:
                result = pd.read_csv(f"/self_repair/cache/mc/{cache_file_name}.csv")
            except FileNotFoundError:
                logger.warning(f"Cache miss on {cache_file_name}, MC will be used")
                data_processed = dc.fromOptimizerToMC(new_configs)
                result = hmtf.run_mc_simulations(data_processed)
                result.to_csv(cache_path, index=False)
                result = dc.fromMCtoOptimizer(result)
        else:
            logger.info(f"Skipping cache for {cache_file_name}, NSGA_II will be used")
            data_processed = dc.fromOptimizerToMC(new_configs)
            result = hmtf.run_mc_simulations(data_processed)
            result.to_csv(f"/self_repair/cache/mc/{cache_file_name}.csv", index=False)
            result = dc.fromMCtoOptimizer(result)
        
        return result

    def opt_optimization(self, new_configs: pd.DataFrame, scs_regressor, cache_file_name: str, skip_caching=False) -> pd.DataFrame:
        cache_dir = "self_repair/cache/opt"
        cache_path = os.path.join(cache_dir, f"{cache_file_name}.csv")

        # Make sure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        if not skip_caching:
            try:
                result = pd.read_csv(cache_path)
                logger.info(f"Cache hit: {cache_file_name}")
            except FileNotFoundError:
                logger.warning(f"Cache miss on {cache_file_name}, Model Checker will be used")
                result = optimize_configurations(new_configs.reset_index(drop=True), scs_regressor)
                result.to_csv(cache_path, index=False)
        else:
            logger.info(f"Skipping cache for {cache_file_name}, Model Checker will be used")
            result = optimize_configurations(new_configs.reset_index(drop=True), scs_regressor)
            result.to_csv(cache_path, index=False)

        return result.drop(columns=["FTG"])
    
class RegressorInterface(MC_OPT_INTERFACE):
    def __init__(self, scs_ground_truth_regressor, skip_cache: bool):
        super().__init__()
        self.scs_ground_truth_regressor = scs_ground_truth_regressor
        self.skip_cache = skip_cache
    
    def mc_results_from_configs(self, new_configs: pd.DataFrame) -> pd.DataFrame:
        scs_y = self.scs_ground_truth_regressor.predict(new_configs)
        new_configs["SCS"] = scs_y
        
        return new_configs

    def opt_optimization(self, new_configs: pd.DataFrame, scs_regressor, cache_file_name: str) -> pd.DataFrame:
        cache_dir = "self_repair/cache/opt"
        cache_path = os.path.join(cache_dir, f"{cache_file_name}.csv")

        # Make sure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        if not self.skip_cache:
            try:
                result = pd.read_csv(cache_path)
                logger.info(f"Cache hit: {cache_file_name}")
            except FileNotFoundError:
                logger.warning(f"Cache miss on {cache_file_name}, NSGA_II will be used")
                result = optimize_configurations(new_configs.reset_index(drop=True), scs_regressor)
                result.to_csv(cache_path, index=False)
        else:
            logger.info(f"Skipping cache for {cache_file_name}, NSGA_II will be used")
            result = optimize_configurations(new_configs.reset_index(drop=True), scs_regressor)
            result.to_csv(cache_path, index=False)

        return result.drop(columns=["FTG"])
