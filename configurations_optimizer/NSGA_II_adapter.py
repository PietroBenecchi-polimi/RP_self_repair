import time
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from tqdm import tqdm
import joblib

all_features = [
    "PRGS", "ORCH_1_Dstop", "ORCH_1_Drestart", "ORCH_1_Fstop", "ORCH_1_Frestart",
    "PSCS__TAU", "HUM_1_VEL", "HUM_2_VEL", "HUM_1_FW", "HUM_1_AGE", "HUM_1_STA",
    "HUM_2_FW", "HUM_2_AGE", "HUM_2_STA", "HUM_1_POS_X", "HUM_1_POS_Y",
    "HUM_2_POS_X", "HUM_2_POS_Y", "ROB_1_VEL", "ROB_1_CHG"
]

feature_names = [
    "ORCH_1_Dstop", "ORCH_1_Drestart", "ORCH_1_Fstop", "ORCH_1_Frestart",
    "HUM_1_VEL", "HUM_2_VEL", "ROB_1_VEL"
]

constant_parameters = [
    "PRGS", "PSCS__TAU", "HUM_1_FW", "HUM_1_AGE", "HUM_1_STA",
    "HUM_2_FW", "HUM_2_AGE", "HUM_2_STA", "HUM_1_POS_X", "HUM_1_POS_Y",
    "HUM_2_POS_X", "HUM_2_POS_Y", "ROB_1_CHG"
]

result_df_columns = all_features + ["SCS", "FTG"]


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["bestSCS"] = []
        self.data["bestFTG"] = []

    def notify(self, algorithm):
        self.data["bestSCS"].append((algorithm.pop.get("F")[:, [0]] * (-1)).mean())
        self.data["bestFTG"].append((algorithm.pop.get("F")[:, [1]]).mean())


class MOO(ElementwiseProblem):
    def __init__(self, row_unmodifiable, regressor_SCS, regressor_FTG, **kwargs):
        super().__init__(n_var=len(feature_names),
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([5.0, 2.0, 0.5, 0.1, 30.0, 30.0, 30.0]),  # Lower bounds for decision variables
                         xu=np.array([7.5, 4.5, 0.8, 0.4, 100.0, 100.0, 100.0]),
                         **kwargs)  # Upper bounds for decision variables

        self.row_unmodifiable = row_unmodifiable
        self.regressor_SCS = regressor_SCS
        self.regressor_FTG = regressor_FTG

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.array(X).reshape(1, 7)
        model_input = self.reorganize_input_indices(X, self.row_unmodifiable)
        success_probability = self.regressor_SCS.predict(model_input)
        muscle_fatigue = self.regressor_FTG.predict(model_input)
        success_probability = -success_probability
        out["F"] = np.array([success_probability, muscle_fatigue])

    def reorganize_input_indices(self, row_modifiable, row_unmodifiable):
        new_df = pd.DataFrame(columns=all_features)
        new_df[feature_names] = row_modifiable
        new_df[constant_parameters] = row_unmodifiable
        return new_df



def process_dataframe(df):
    freewill_mapping = {"foc": 0, "distr": 1, "free": 2}
    age_mapping = {"y": 0, "e": 1}
    health_mapping = {"h": 0, "s": 1, "u": 2}

    transformations = {
        'HUM_1_FW': freewill_mapping,
        'HUM_2_FW': freewill_mapping,
        'HUM_1_AGE': age_mapping,
        'HUM_2_AGE': age_mapping,
        'HUM_1_STA': health_mapping,
        'HUM_2_STA': health_mapping
    }

    df = df.copy()
    for col, mapping in transformations.items():
        df[col] = df[col].replace(mapping)
    return df


def optimize_configurations(
    df,
    regressor_SCS,
    pop_size=20,
    n_gen=20
):
    df = process_dataframe(df)

    # Check for missing values
    if df.isnull().values.any():
        raise ValueError("Input DataFrame contains NaN values. Please clean the data.")

    regressor_FTG = joblib.load("./configurations_optimizer/regressor_FTG.joblib")
    time_records = []
    results = []

    val_SCS_averaged = []
    val_FTG_averaged = []
    unfeasible_configurations = 0

    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=df.shape[0]):
        start_time = time.time()

        problem = MOO(
            df[constant_parameters].to_numpy()[idx].reshape((1, len(constant_parameters))),
            regressor_SCS,
            regressor_FTG
        )
        algorithm = NSGA2(pop_size=pop_size)
        termination = ("n_gen", n_gen)

        res = minimize(problem, algorithm, termination=termination, seed=1,
                       save_history=True, callback=MyCallback(), verbose=False)

        end_time = time.time()
        valSCS = res.algorithm.callback.data["bestSCS"]
        valFTG = res.algorithm.callback.data["bestFTG"]
        val_SCS_averaged.append(valSCS)
        val_FTG_averaged.append(valFTG)

        # Assemble results cleanly using a dictionary
        result_row = {
            **{name: val for name, val in zip(feature_names, res.X[-1])},
            **{name: df.loc[idx, name] for name in constant_parameters},
            "SCS": -res.F[-1, 0],
            "FTG": res.F[-1, 1]
        }
        results.append(result_row)

        iteration_duration = end_time - start_time
        if iteration_duration > df.loc[idx, "PSCS__TAU"]:
            unfeasible_configurations += 1
        time_records.append({
            "Iteration_duration": iteration_duration,
            "PSCS__TAU": df.loc[idx, "PSCS__TAU"]
        })

    result_df = pd.DataFrame(results, columns=result_df_columns)

    return result_df