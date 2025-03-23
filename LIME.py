import pandas as pd
import numpy as np
from lime import lime_tabular
import joblib
import matplotlib.pyplot as plt
import ast
import os
import numpy as np

def save_lime_explanation_plot(explanation, instance_index, output_dir):
    """
    Saves the LIME explanation plot for a given instance.

    Parameters:
    - explanation: The LIME explanation object.
    - instance_index: The index of the instance being explained.
    - output_dir: The directory where the plot will be saved.
    """
    if output_dir:
        print(f"Saving explanation plot for instance {instance_index}...")
        plt.figure(figsize=(10, 6))
        explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.title(f"LIME Explanation for Instance {instance_index}")
        plt.savefig(os.path.join(output_dir, f"lime_explanation_{instance_index}.png"))
        plt.close()

def explain_prediction_with_lime(csv_path, model_path, num_features, plot_explanations=False):
    # Create path 
    output_dir = "lime_explanations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    df = pd.read_csv(csv_path)
    # Convert the string representation of the dictionary to a dictionary
    df["opt_config"] = df["opt_config"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df = df["opt_config"].apply(pd.Series)

    X = df.drop(columns=['SCS', 'FTG'])
    y = df['SCS']
    # Get feature names
    feature_names = X.columns.tolist()
    model = joblib.load(model_path)
    model.predict(X)
    print(f"Loaded model from: {model_path}")

    # Create a prediction function that returns the prediction of the model
    def predict_fn(instances):
        return model.predict(pd.DataFrame(instances, columns=feature_names))

    explainer = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X), # Training dataset without target
        mode = "regression", # "classification" or "regression"
        feature_names=feature_names, # Feature names
    )

    explanations = []
    explanation_dicts = []
    for instance_index in range(0, max(0, len(X))):
        print(f"\nExplaining instance {instance_index + 1}/{len(X)}...")
        
        # Get the instance to explain
        instance_to_explain = X.iloc[instance_index].values
        
        # Generate explanation
        explanation = explainer.explain_instance(
            data_row=instance_to_explain,
            predict_fn=predict_fn,
            num_features=num_features
        )
        
        # Store explanation
        explanations.append(explanation)
        
        # Display the feature contributions: Can be cancelled
        print(f"Feature contributions for instance {instance_index}:")
        explanation_dict = {}
        for feature, weight in explanation.as_list():
            feature_name = [name for name in feature.split() if name in feature_names][0]
            explanation_dict[feature_name] = weight
        explanation_dicts.append(explanation_dict)
        # Save plot if requested
        if plot_explanations:
            save_lime_explanation_plot(explanation, instance_index, output_dir)
    return explanation_dicts

if __name__ == "__main__":
    file_name = "invalid_config_0.17_0.01"
    csv_path = f"./Matching_and_verifier/invalid_configs/{file_name}.csv"
    model_path = "regressor_SCS.joblib"
    
    explainations = explain_prediction_with_lime(csv_path, model_path, num_features=20, plot_explanations=False)
    explainations_df = pd.DataFrame(explainations)
    explainations_df.to_csv(f"./lime_explanations/{file_name}.csv")