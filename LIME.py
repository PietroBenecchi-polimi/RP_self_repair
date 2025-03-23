import pandas as pd
import numpy as np
from lime import lime_tabular
import joblib
import matplotlib.pyplot as plt
import ast
import os

def explain_prediction_with_lime(csv_path, model_path, num_features):
    # Crea la cartella se non esiste già
    output_dir = "lime_explanations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Directory corrente:", os.getcwd())
    # Load the dataset
    df = pd.read_csv(csv_path)
    # Convertire la colonna 'opt_config' da stringa a dizionario
    df["opt_config"] = df["opt_config"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Espandere il dizionario in colonne separate
    df = df["opt_config"].apply(pd.Series)
    X = df.drop(columns=['SCS', 'FTG'])
    y = df['SCS']
    # Get feature names
    feature_names = X.columns.tolist()
    model = joblib.load(model_path)
    model.predict(X)
    print(f"Loaded model from: {model_path}")

    # Create a prediction function that returns the right shape
    def predict_fn(instances):
        return model.predict(pd.DataFrame(instances, columns=feature_names))
    # https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=limetabularexplainer#lime.lime_tabular.LimeTabularExplainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X), # Dataset di addestramento senza target
        mode = "regression", # "classification" o "regression"
        feature_names=feature_names, # Nomi delle feature
    )

    explanations = []
    for instance_index in range(1):
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
        
        # Display the feature contributions
        print(f"Feature contributions for instance {instance_index}:")
        for feature, weight in explanation.as_list():
            print(f" {feature}: {weight:.4f}")
        
        # Plot the explanation if requested
        if output_path:
            print(f"Saving explanation plot for instance {instance_index}...")
            plt.figure(figsize=(10, 6))
            explanation.as_pyplot_figure()
            plt.tight_layout()
            plt.title(f"LIME Explanation for Instance {instance_index}")
            plt.savefig(os.path.join(output_dir, f"lime_explanation_{instance_index}.png"))
            plt.close()

    return explanations

if __name__ == "__main__":
    csv_path = "ImprovedDataToMCMatches/match_results/matches_0.1.csv"
    model_path = "regressor_SCS.joblib"
    # Mostrare i primi risultati
    explain_prediction_with_lime(csv_path, model_path, num_features=20)