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

        epsilon = abs(opt_SCS - mc_SCS)
        return epsilon < 0.1, epsilon

    def __init__(self, training_dataset_path: str, test_data_path: str, points_regressor: int, n_data_to_verify: int):
        dataset = ut.load_dataset_for_regressor(training_dataset_path)
        self.ground_truth_regressor = self.__train_new_regressor(dataset)
        self.train_data = dataset.sample(points_regressor, random_state=128).reset_index(drop=True)
        self.regressor = self.__train_new_regressor(self.train_data)
        # It is a small set based on test_data_path. The size is determined by n_data_to_verify.
        self.test_set = ut.load_dataset_for_regressor(test_data_path).sample(n_data_to_verify, random_state=128).reset_index(drop=True)
        
    def validate_configurations(self, opt_results: pd.DataFrame, ground_truth):
        invalid_results = []
        epsilon_array = {}

        # **Iterate through results and validate configurations**
        for idx, result in opt_results.iterrows():
            opt_SCS = result['SCS']
            mc_SCS = ground_truth.iloc[idx]['SCS']

            # Validate SCS
            valid, epsilon = self.__validation_metric(opt_SCS, mc_SCS)

            if not valid:
                invalid_results.append(result)

            epsilon_array[idx] = epsilon

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
            PlugInvalid
        ]

        results_dict = {}

        def run_method(cls: OversamplingMethod):
            if cls == LimeBasedOversampling:
                instance: OversamplingMethod = cls(self.regressor)
                instance.run_oversampling(df=invalid_configurations.copy(), n_samples=n_samples)
            elif issubclass(cls, SmoteBasedOversampling):
                instance: SmoteBasedOversampling = cls(invalid_configurations.copy())
                instance.run_oversampling(n_samples=n_samples)
            elif issubclass(cls, PlugInvalid):
                instance: PlugInvalid = cls()
                instance.run_oversampling(df=invalid_configurations.copy())
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
        combined_dataset = pd.concat([oversampling.copy(), self.train_data.copy()], ignore_index=True)
        X = combined_dataset.drop(columns=["SCS"])
        y = combined_dataset["SCS"]

        regressor_copy = clone(self.regressor)
        regressor_copy.fit(X, y)
        
        return regressor_copy
    
    def generate_neighbours_from_config(self, config: pd.DataFrame, neighbours_to_generate=20, offset=0.025, regressor = None):
            with open('data/neighbours_factor.json', 'r') as file:
                factors = dict(json.load(file))

            neighbours = pd.DataFrame()
            
            transformation_rules = ut.get_transformation_rules()
            regressor = regressor if regressor is not None else self.regressor

            # If target_config is a DataFrame with one row, convert to Series for easier access
            if len(config) != 1:
                raise ValueError("config should be a DataFrame with exactly one row.")
            config = config.iloc[0]

            for factor_key in config.index:
                # for categorical features, we just repeat the target value
                if factor_key in transformation_rules:
                    neighbours[factor_key] = [config[factor_key]] * neighbours_to_generate
                elif factor_key in factors:
                    factor_info = factors[factor_key]
                    # Determine type and range
                    col_min = factor_info.get("min")
                    col_max = factor_info.get("max")
                    is_int = factor_info.get("type", "float") == "int"
                    # If no min/max, fallback to target value ±offset
                    center = config[factor_key]
                    # sample around the target value, but within allowed range
                    spread = (col_max - col_min) * offset
                    low = max(col_min, center - spread)
                    high = min(col_max, center + spread)

                    if is_int:
                        if low >= high:
                            neighbours[factor_key] = [int(center)] * neighbours_to_generate
                        else:
                            neighbours[factor_key] = np.random.randint(np.ceil(low), np.floor(high) + 1, neighbours_to_generate)
                    else:
                        neighbours[factor_key] = np.random.uniform(low, high, neighbours_to_generate)

            neighbours['SCS'] = regressor.predict(neighbours)
            
            return neighbours
