import pandas as pd
import numpy as np
import json
import smogn
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from self_repair.oversampling_methods.LIME import explain_prediction_with_lime
from utils.datacleaner import get_transformation_rules
import utils.datacleaner as dc

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
                mean = float(new_sample[feature].values[0])
                importance = explanations.loc[index].get(feature, 0.0)
                variance = np.clip(abs(1.0 / (importance + epsilon)), 0.01, 1.0)

                if feature in self.transformation_rules:
                    values = np.array(list(self.transformation_rules[feature].values()), dtype=float)
                    probabilities = np.exp(-0.5 * ((values - mean) / variance) ** 2)
                    probabilities /= probabilities.sum()
                    new_value = np.random.choice(values, p=probabilities)
                else:
                    if feature == "PRGS":
                        values = np.array([0, 1, 2, 3, 4, 5])
                        probabilities = np.exp(-0.5 * ((values - mean) / variance) ** 2)
                        probabilities /= probabilities.sum()
                        new_value = np.random.choice(values, p=probabilities)
                    elif feature == "PSCS__TAU":
                        new_value = np.random.choice(np.arange(250, 751))
                    else:
                        if feature in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                            col_max, col_min = factors["HUM_1_POS"]["max_x"], factors["HUM_1_POS"]["min_x"]
                        elif feature in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                            col_max, col_min = factors["HUM_1_POS"]["max_y"], factors["HUM_1_POS"]["min_y"]
                        else:
                            col_max, col_min = factors[feature]["max"], factors[feature]["min"]
                        new_value = np.random.uniform(col_min, col_max)

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

        for feature in df.columns:
            if feature not in self.transformation_rules:
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
                kde.fit(df[[feature]])
                feature_samples[feature] = kde.sample(n_samples).flatten()
            else:
                values = np.array(list(self.transformation_rules[feature].values()), dtype=float)
                feature_samples[feature] = np.random.choice(values, n_samples)

        new_samples = pd.DataFrame(feature_samples)
        self.setResampling(dc.castIntegerFeatures(new_samples))

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

class ADASYNOversampling(SmoteBasedOversampling):
    def __init__(self, unbalanced_dataset: pd.DataFrame = pd.DataFrame()):
        super().__init__(unbalanced_dataset)
        self.name_id = "ADASYN"

    def run_oversampling(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        df = df.sample(n_samples) if len(df) > n_samples else df
        df.reset_index(drop=True)
        X = df.drop(columns=["SCS"])
        y = df["SCS"]
        y_class = (y > 0.5).astype(int)

        ada = ADASYN(sampling_strategy='minority', random_state=42)
        X_res, y_res = ada.fit_resample(X, y_class)

        df_resampled = X_res.copy()
        df_resampled["SCS"] = y_res
        self.setResampling(df_resampled)

class BorderlineSMOTEOversampling(SmoteBasedOversampling):
    def __init__(self, unbalanced_dataset: pd.DataFrame = pd.DataFrame()):
        super().__init__(unbalanced_dataset)
        self.name_id = "BorderlineSMOTE"

    def run_oversampling(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        df = df.sample(n_samples) if len(df) > n_samples else df
        df.reset_index(drop=True)
        X = df.drop(columns=["SCS"])
        y = df["SCS"]
        y_class = (y > 0.5).astype(int)

        sm = BorderlineSMOTE(sampling_strategy='minority', random_state=42)
        X_res, y_res = sm.fit_resample(X, y_class)

        df_resampled = X_res.copy()
        df_resampled["SCS"] = y_res
        self.setResampling(df_resampled)

class ClusterSMOTEOversampling(SmoteBasedOversampling):
    def __init__(self, unbalanced_dataset: pd.DataFrame = pd.DataFrame()):
        super().__init__(unbalanced_dataset)
        self.name_id = "ClusterSMOTE"

    def run_oversampling(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        df = df.sample(n_samples) if len(df) > n_samples else df
        df.reset_index(drop=True)
        X = df.drop(columns=["SCS"])
        y = df["SCS"]
        y_class = (y > 0.5).astype(int)

        X = X.copy()
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        X["cluster"] = clusters

        dfs = []
        for cl in np.unique(clusters):
            sub_df = X[X["cluster"] == cl].drop(columns=["cluster"])
            sub_y = y_class[X["cluster"] == cl]

            if sub_y.nunique() < 2:
                continue

            sm = SMOTE(sampling_strategy='minority', random_state=42)
            X_res, y_res = sm.fit_resample(sub_df, sub_y)

            tmp_df = X_res.copy()
            tmp_df["SCS"] = y_res
            dfs.append(tmp_df)

        self.setResampling(pd.concat(dfs, ignore_index=True))