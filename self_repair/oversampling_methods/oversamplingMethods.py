import pandas as pd
import numpy as np
import json
import smogn
from sklearn.neighbors import KernelDensity
from self_repair.oversampling_methods.LIME import explain_prediction_with_lime
from utils.datacleaner import get_transformation_rules
import utils.datacleaner as dc
from utils.rp_logger import logger
with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

class OversamplingMethod:
    def __init__(self):
        self._resampling = pd.DataFrame.empty
        self.transformation_rules = get_transformation_rules()
        self.name_id = "Base"

    @property
    def resampling(self):
        return self._resampling

    @resampling.setter
    def resampling(self, value):
        self._resampling = value

    def getResampling(self):
        return self._resampling

    def getNameId(self):
        return self.name_id

    def setResampling(self, resampling):
        self.resampling = resampling
        
    def run_oversampling(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        pass

class RandomOversampling(OversamplingMethod):
    def __init__(self):
        super().__init__()
        self.name_id = "Random"

    def run_oversampling(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        synthetic_data = {}

        for factor_key in df.columns:
            if factor_key == "SCS":
                synthetic_data[factor_key] = np.random.uniform(0, 1, n_samples)
            elif factor_key not in self.transformation_rules:
                if factor_key == "PRGS":
                    synthetic_data[factor_key] = np.random.choice([0, 1, 2, 3, 4, 5], n_samples)
                elif factor_key == "PSCS__TAU":
                    synthetic_data[factor_key] = np.random.choice(np.arange(250, 751), n_samples)
                else:
                    if factor_key in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                        col_max, col_min = factors["HUM_1_POS"]["max_x"], factors["HUM_1_POS"]["min_x"]
                    elif factor_key in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                        col_max, col_min = factors["HUM_1_POS"]["max_y"], factors["HUM_1_POS"]["min_y"]
                    else:
                        col_max, col_min = factors[factor_key]["max"], factors[factor_key]["min"]
                    synthetic_data[factor_key] = np.random.uniform(col_min, col_max, n_samples)
            else:
                values = list(self.transformation_rules[factor_key].values())
                synthetic_data[factor_key] = np.random.choice(values, n_samples)

        self.setResampling(pd.DataFrame(synthetic_data))

class LimeBasedOversampling(OversamplingMethod):
    def __init__(self, regressor):
        super().__init__()
        self.regressor = regressor
        self.name_id = "LimeBased"

    def run_oversampling(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        df = df.drop(columns=["SCS"], errors='ignore').reset_index(drop=True)
        new_samples = []
        epsilon = 1e-5

        explanations = explain_prediction_with_lime(df, self.regressor, num_features=20)

        for _ in range(n_samples):
            index = df.sample(n=1).index[0]
            new_sample = df.loc[[index]].copy()

            for feature in new_sample.columns:

                if feature in self.transformation_rules:
                    continue
                else:
                    mean = float(new_sample[feature].values[0])
                    importance = explanations.loc[index].get(feature, 0.0)
                    variance = np.clip(abs(1.0 / (importance + epsilon)), 0.01, 1.0)
                    if feature == "PRGS":
                        values = np.array([0, 1, 2, 3, 4, 5])
                        probabilities = np.exp(-0.5 * ((values - mean) / variance) ** 2)
                        probabilities /= probabilities.sum()
                        new_value = np.random.choice(values, p=probabilities)
                    elif feature == "PSCS__TAU":
                        new_value = np.random.choice(np.arange(250, 751))
                    else:
                        new_value = np.random.normal(mean, variance)

                new_sample[feature] = new_value
            new_samples.append(new_sample)

        self.setResampling(pd.concat(new_samples, ignore_index=True))

class KDEOversampling(OversamplingMethod):
    def __init__(self):
        super().__init__()
        self.name_id = "KDE"

    def run_oversampling(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        df = df.drop(columns=["SCS"], errors='ignore')
        feature_samples = {}
        n_candidates = 10000

        for feature in df.columns:
            if feature not in self.transformation_rules:
                data = df[[feature]].values

                # Fit KDE
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
                kde.fit(data)

                # Define candidate range
                min_val, max_val = data.min(), data.max()
                candidates = np.linspace(min_val, max_val, n_candidates).reshape(-1, 1)

                # Evaluate density
                log_dens = kde.score_samples(candidates)
                density = np.exp(log_dens)

                # Avoid division by zero and normalize inverse density
                inv_density = 1 / (density + 1e-10)
                inv_density /= inv_density.sum()

                # Sample from inverse density
                rng = np.random.default_rng(seed=128)
                sampled_indices = rng.choice(n_candidates, size=n_samples, p=inv_density)
                sampled_points = candidates[sampled_indices].flatten()
                feature_samples[feature] = sampled_points
            else:
                feature_samples[feature] = df[feature].sample(n=n_samples, replace=True).values

        new_samples = pd.DataFrame(feature_samples)
        self.setResampling(dc.castIntegerFeatures(new_samples))

class PlugInvalid(OversamplingMethod):

    def __init__(self):
        super().__init__()
        self.name_id = "PlugInvalid"
        
    def run_oversampling(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["SCS"], errors='ignore')
        self.setResampling(dc.castIntegerFeatures(df))

class SmoteBasedOversampling(OversamplingMethod):
    def __init__(self, unbalanced_dataset: pd.DataFrame = pd.DataFrame()):
        super().__init__()
        self.unbalanced_dataset = unbalanced_dataset

class SmoteOversampling(SmoteBasedOversampling):
    def __init__(self, unbalanced_dataset: pd.DataFrame):
        super().__init__(unbalanced_dataset)
        self.name_id = "Smote"

    def run_oversampling(self, n_samples: int) -> pd.DataFrame:
        df = self.unbalanced_dataset.sample(n_samples) if len(self.unbalanced_dataset) > n_samples else self.unbalanced_dataset
        df.reset_index(drop=True)
        df_resampled = smogn.smoter(
            data=df,
            y='SCS',
            k=5,
            rel_coef=0.35,
            rel_thres=df['SCS'].quantile(0.9)
        )
        self.setResampling(dc.castIntegerFeatures(df_resampled))