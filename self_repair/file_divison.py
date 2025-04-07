import pandas as pd

# Read the dataset
data = pd.read_csv("datasets/configurations_improved_20_20.csv")

# Shuffle the dataset
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Get the first 100 rows
training_data_before_oversampling = shuffled_data.head(100)
training_data_before_oversampling.to_csv("datasets/first_100_rows.csv", index=False)

# Get the last 100 rows
test_data = shuffled_data.tail(100)
test_data.to_csv("datasets/last_100_rows.csv", index=False)

# Get the remaining rows for training
training_data = shuffled_data.iloc[100:-100]
training_data.to_csv("datasets/training_data.csv", index=False)