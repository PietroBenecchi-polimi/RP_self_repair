import pandas as pd
import numpy as np
from lime import lime_tabular
import joblib
import matplotlib.pyplot as plt
import ast
import os
import numpy as np
import sys
warnings.simplefilter("ignore", InconsistentVersionWarning)

sys.path.append(os.path.abspath("utils"))
from rp_logger import logger
def save_lime_explanation_plot(explanation, instance_index, output_dir):
    """
    Saves the LIME explanation plot for a given instance.

    Parameters:
    - explanation: The LIME explanation object.
    - instance_index: The index of the instance being explained.
    - output_dir: The directory where the plot will be saved.
    """
    if output_dir:
        logger.debug(f"Saving explanation plot for instance {instance_index}...")
        plt.figure(figsize=(10, 6))
        explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.title(f"LIME Explanation for Instance {instance_index}")
        plt.savefig(os.path.join(output_dir, f"lime_explanation_{instance_index}.png"))
        plt.close()

def explain_prediction_with_lime(df, model, num_features):
    # Get feature names
    feature_names = df.columns.tolist()
    model.predict(df)

    # Create a prediction function that returns the prediction of the model
    def predict_fn(instances):
        return model.predict(pd.DataFrame(instances, columns=feature_names))

    explainer = lime_tabular.LimeTabularExplainer(
        training_data = np.array(df), # Training dataset without target
        mode = "regression", # "classification" or "regression"
        feature_names=feature_names, # Feature names
    )

    explanations = []
    explanation_dicts = []
    for instance_index in range(0, max(0, len(df))):        
        # Get the instance to explain
        instance_to_explain = df.iloc[instance_index].values
        
        # Generate explanation
        explanation = explainer.explain_instance(
            data_row=instance_to_explain,
            predict_fn=predict_fn,
            num_features=num_features
        )
        
        # Store explanation
        explanations.append(explanation)
        
        # Display the feature contributions: Can be cancelled
        explanation_dict = {}
        for feature, weight in explanation.as_list():
            feature_name = [name for name in feature.split() if name in feature_names][0]
            explanation_dict[feature_name] = weight
        explanation_dicts.append(explanation_dict)

    explanation_dicts = pd.DataFrame(explanation_dicts)
    return explanation_dicts

if __name__ == "__main__":
    file_name = sys.argv[0]
    csv_path = f"./Matching_and_verifier/invalid_configs/{file_name}.csv"
    model_path = "regressor_SCS.joblib"
    
    explainations = explain_prediction_with_lime(csv_path, model_path, num_features=20, plot_explanations=False)
    explainations_df = pd.DataFrame(explainations)
    explainations_df.to_csv(f"./lime_explanations/{file_name}.csv")