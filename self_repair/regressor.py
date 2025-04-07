import joblib
import pandas as pd
from validation import validate_configurations

regressor = joblib.load("self_repair/regressor/regressor_SCS.joblib")
new_data = pd.read_csv("datasets/configurations_improved_20_20.csv")
new_data = new_data.drop(columns=["SCS", "FTG"])

new_data['SCS'] = regressor.predict(new_data)
ground_truth = pd.read_csv("datasets/configurations_improved_20_20.csv")

invalid_results_df, success_percentage = validate_configurations(new_data, ground_truth, FTG_threshold=0.01)
print(f"Success percentage: {success_percentage}")