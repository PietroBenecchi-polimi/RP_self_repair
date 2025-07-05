import pandas as pd
import json
import utils.datacleaner as ut
import numpy as np
import self_repair.pipeline as pipeline
import visualization.visualization as vis
import utils.datacleaner as dc

def generate_neighbours_from_config(config: pd.DataFrame, neighbours_to_generate=20, offset=0.025, regressor = None):
        with open('data/neighbours_factor.json', 'r') as file:
            factors = dict(json.load(file))

        neighbours = pd.DataFrame()
        
        transformation_rules = ut.get_transformation_rules()

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

        return neighbours

def main():
    invalid_stats = pd.read_pickle('visualization/data/data_50_configs_mc/oversampling_results_invalid_configs.pkl')
    df_invalid = dc.process_results(invalid_stats)
    df_invalid['Validation Type'] = 'Invalid Configs'
    
    vis.plot_allConfigs_boxplot(df_invalid, test_name="50_configs_mc")

if __name__ == "__main__":
    main()