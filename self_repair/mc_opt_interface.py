import pandas as pd
from utils.rp_logger import logger
import model_checker.hri_designtime.src.hmt_factors as hmtf
from configurations_optimizer.NSGA_II_adapter import optimize_configurations
import utils.datacleaner as dc
from abc import ABC, abstractmethod


class MC_OPT_INTERFACE(ABC):
    @abstractmethod
    def mc_results_from_configs(self, new_configs: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def opt_optimization(self, new_configs: pd.DataFrame, scs_regressor) -> pd.DataFrame:
        pass
    @abstractmethod
    def setgroundTruth(self, new_configs: pd.DataFrame, scs_regressor) -> pd.DataFrame:
        pass


class ModelCheckerInterface(MC_OPT_INTERFACE):
    def mc_results_from_configs(self, new_configs: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running model checker simulations (no cache)...")
        data_processed = dc.fromOptimizerToMC(new_configs)
        result = hmtf.run_mc_simulations(data_processed)
        return dc.fromMCtoOptimizer(result)

    def opt_optimization(self, new_configs: pd.DataFrame, scs_regressor) -> pd.DataFrame:
        logger.info("Running NSGA-II optimization (no cache)...")
        result = optimize_configurations(new_configs.reset_index(drop=True), scs_regressor)
        return result.drop(columns=["FTG"])
    def setgroundTruth(self, scs_regressor) -> pd.DataFrame:
        pass


class RegressorInterface(MC_OPT_INTERFACE):
    def __init__(self):
        pass

    def mc_results_from_configs(self, new_configs: pd.DataFrame) -> pd.DataFrame:
        logger.info("Predicting with regressor (no cache)...")
        scs_y = self.scs_ground_truth_regressor.predict(new_configs)
        new_configs = new_configs.copy()
        new_configs["SCS"] = scs_y
        return new_configs

    def opt_optimization(self, new_configs: pd.DataFrame, scs_regressor) -> pd.DataFrame:
        logger.info("Running NSGA-II optimization (no cache)...")
        result = optimize_configurations(new_configs.reset_index(drop=True), scs_regressor)
        return result.drop(columns=["FTG"])
    
    def setgroundTruth(self, scs_regressor) -> pd.DataFrame:
        self.scs_ground_truth_regressor = scs_regressor