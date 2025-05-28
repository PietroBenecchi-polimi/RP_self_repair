import utils.datacleaner as ut
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from self_repair.oversamplingMethods import *
from sklearn.base import clone

class Pipeline:
    ground_truth_regressor = None
    regressor = None
    verification_set = []
    resampling_sets = []
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

    def __init__(self, training_dataset_path: str, verification_data_path: str, points_regressor: int, n_data_to_verify: int):
        dataset = ut.load_dataset_for_regressor(training_dataset_path)
        Pipeline.ground_truth_regressor = self.__train_new_regressor(dataset)
        self.initial_dataset = dataset.sample(points_regressor, random_state=128).reset_index(drop=True)
        Pipeline.regressor = self.__train_new_regressor(self.initial_dataset)
        Pipeline.verification_dataset = ut.load_dataset_for_regressor(verification_data_path).sample(n_data_to_verify, random_state=128).reset_index(drop=True)

    def validate_configurations(self, opt_results, ground_truth):
        validity_array = []
        epsilon_array = []
        invalid_results = []

        # **Iterate through results and validate configurations**
        for i in range(len(opt_results)):
            opt_SCS = opt_results.iloc[i]['SCS']
            try:
                mc_SCS = ground_truth.iloc[i]['SCS']
            except IndexError:
                continue
            # Validate SCS
            validity, epsilon = self.__validation_metric(opt_SCS, mc_SCS)

            if not validity:
                invalid_results.append(opt_results.iloc[i])

            validity_array.append(validity)
            epsilon_array.append(epsilon)

        invalid_results_df = pd.DataFrame(invalid_results)

        return invalid_results_df, epsilon_array

    def oversample(self, n_samples: int, invalid_configurations: pd.DataFrame):
        methods = [
            RandomOversampling,
            LimeBasedOversampling,
            KDEOversampling,
            SmoteOversampling,
            ADASYNOversampling,
            BorderlineSMOTEOversampling,
            ClusterSMOTEOversampling
        ]

        results_dict = {}

        def run_method(cls: OversamplingMethod):
            if cls == LimeBasedOversampling:
                instance: OversamplingMethod = cls(self.regressor)
                instance.run_oversampling(df=self.initial_dataset.copy(), n_samples=n_samples)
            elif issubclass(cls, SmoteBasedOversampling):
                instance: SmoteBasedOversampling = cls(invalid_configurations)
                instance.run_oversampling(n_samples=n_samples)
            else:
                instance: OversamplingMethod = cls()
                instance.run_oversampling(df=self.initial_dataset.copy(), n_samples=n_samples)
            return instance.name_id, instance.getResampling()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_method, cls) for cls in methods]
            for future in futures:
                name, resampled_df = future.result()
                results_dict[name] = resampled_df

        return results_dict

    
    def retrain_regressor(self, oversampling: pd.DataFrame):
        combined_dataset = pd.concat([oversampling, self.initial_dataset], ignore_index=True)
        X = combined_dataset.drop(columns=["SCS"])
        y = combined_dataset["SCS"]

        regressor_copy = clone(self.regressor)
        regressor_copy.fit(X, y)

        return regressor_copy