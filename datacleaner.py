import pandas as pd

# Load the datasets
df1 = pd.read_csv('./datasets/initial_configurations_to_improve.csv')
df2 = pd.read_csv('./datasets/configurations_improved_20_20.csv')

# Define the transformation rules for categorical columns
transform_rules = {
    'HUM_1_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_1_AGE': {'y': 0.0, 'e': 1.0},
    'HUM_1_STA': {'s': 1.0, 'h': 0.0, 'u': 2.0},
    'HUM_2_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_2_AGE': {'y': 0.0, 'e': 1.0},
    'HUM_2_STA': {'s': 1.0, 'h': 0.0, 'u': 2.0}
}

# Apply the transformation rules to the first dataset
for column, mapping in transform_rules.items():
    df1[column] = df1[column].map(mapping)

# Ensure the columns in df1 match the columns in df2
# Add missing columns to df1 with default values (e.g., 0.0)
for column in df2.columns:
    if column not in df1.columns:
        df1[column] = 0.0  # Default value for missing columns

# Remove extra columns from df1 that are not in df2
df1 = df1[df2.columns]

# Save the transformed dataset to a new CSV file
df1.to_csv('./ImprovedDataToMCMatches/refinedData/transformed_dataset.csv', index=False)

print("Transformation complete. The transformed dataset is saved as 'transformed_dataset.csv'.")