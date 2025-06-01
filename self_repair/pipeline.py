import utils.datacleaner as ut
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from self_repair.oversampling_methods.oversamplingMethods import *
from sklearn.base import clone

class Pipeline:
    
    @classmethod
    def __train_new_regressor(cls, training_set: pd.DataFrame):
        X_train = training_set.drop(columns=["SCS"])
        y_train = training_set["SCS"]

        regressor = RandomForestRegressor(random_state=42)
        regressor.fit(X_train, y_train)

        return regressor
    
    @classmethod
    def __validation_metric(cls, opt_SCS, mc_SCS):
        epsilon = np.abs(opt_SCS - mc_SCS)
        return epsilon < 0.1, epsilon

    def __init__(self, training_dataset_path: str, test_data_path: str, points_regressor: int, n_data_to_verify: int):
        dataset = ut.load_dataset_for_regressor(training_dataset_path)
        self.ground_truth_regressor = self.__train_new_regressor(dataset)
        self.train_data = dataset.sample(points_regressor, random_state=128).reset_index(drop=True)
        self.regressor = self.__train_new_regressor(self.train__data)
        # It is a small set based on test_data_path. The size is determined by n_data_to_verify.
        self.test_set = ut.load_dataset_for_regressor(test_data_path).sample(n_data_to_verify, random_state=128).reset_index(drop=True)
        
    def validate_configurations(self, opt_results, ground_truth):
        epsilon_array = []
        invalid_results = []

        # **Iterate through results and validate configurations**
        for i in range(len(opt_results)):
            opt_SCS = opt_results.iloc[i]['SCS']
            mc_SCS = ground_truth.iloc[i]['SCS']

            # Validate SCS
            valid, epsilon = self.__validation_metric(opt_SCS, mc_SCS)

            if not valid:
                invalid_results.append(opt_results.iloc[i])

            epsilon_array.append(epsilon)

        invalid_results_df = pd.DataFrame(invalid_results)

        return invalid_results_df, epsilon_array

    ## Run all oversampling methods in parallel. Is it parallel? 
    ## Output: dictionary <method_name, new_data>
    ## IMPORTANT: data is generated without metrics(SCS, FTG)
    def oversample(self, n_samples: int, invalid_configurations: pd.DataFrame):
        methods = [
            RandomOversampling,
            LimeBasedOversampling,
            KDEOversampling,
            #SmoteOversampling,
            ADASYNOversampling,
            BorderlineSMOTEOversampling,
            ClusterSMOTEOversampling
        ]

        results_dict = {}

        def run_method(cls: OversamplingMethod):
            if cls == LimeBasedOversampling:
                instance: OversamplingMethod = cls(self.regressor)
                instance.run_oversampling(df=self.test_set.copy(), n_samples=n_samples)
            elif issubclass(cls, SmoteBasedOversampling):
                instance: SmoteBasedOversampling = cls(invalid_configurations)
                instance.run_oversampling(df=self.test_set, n_samples=n_samples)
            else:
                instance: OversamplingMethod = cls()
                instance.run_oversampling(df=self.test_set.copy(), n_samples=n_samples)
            return instance.name_id, instance.getResampling()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_method, cls) for cls in methods]
            for future in futures:
                name, resampled_df = future.result()
                results_dict[name] = resampled_df

        return results_dict
    
    def retrain_regressor(self, oversampling: pd.DataFrame):
        self.train_data = pd.concat([oversampling, self.train_data], ignore_index=True)
        X = self.train_data.drop(columns=["SCS"])
        y = self.train_data["SCS"]

        regressor_copy = clone(self.regressor)
        regressor_copy.fit(X, y)
        self.regressor = regressor_copy
        
        return self.regressor