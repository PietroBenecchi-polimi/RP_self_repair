import json
from utils.datacleaner import get_transformation_rules
import numpy as np
import pandas as pd

with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

## Comment TODO: Mi sembra non abbia molto senso questa funzione. Io devo generare nell'intorno del punto preso
def synthesize_data(target_config: pd.DataFrame, data_to_generate = 20):
    synthetic_data = {}

    transformation_rules = get_transformation_rules()

    for factor_key in target_config.columns:
        if factor_key == "SCS":
            synthetic_data[factor_key] = np.random.uniform(0, 1, data_to_generate)
        elif factor_key not in transformation_rules.keys():
            if "PRGS" == factor_key:
                synthetic_data[factor_key] = np.random.choice([0,1,2,3,4,5])
            else:
                if factor_key in ["HUM_1_POS_X", "HUM_2_POS_X"]:
                    col_max, col_min = factors["HUM_1_POS"]["max_x"], factors["HUM_1_POS"]["min_x"]
                elif factor_key in ["HUM_1_POS_Y", "HUM_2_POS_Y"]:
                    col_max, col_min = factors["HUM_1_POS"]["max_y"], factors["HUM_1_POS"]["min_y"]
                elif "max" in factors[factor_key]:
                    col_max, col_min = factors[factor_key]["max"], factors[factor_key]["min"]

                synthetic_data[factor_key] = np.random.uniform(col_min, col_max, data_to_generate)
        else:
            values = list(transformation_rules[factor_key].values())
            synthetic_data[factor_key] = np.random.choice(values, data_to_generate)

    return pd.DataFrame(synthetic_data)
