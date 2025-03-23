import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from LIME import explain_prediction_with_lime 
import re

def extract_feature_importances(explanations):
    all_importances = {}

    for i, exp in enumerate(explanations):
        # extract feature feature_weights (feature interval, weight)
        feature_weights = exp.as_list()
        
        # create a dictionary of feature importances
        importance_dict = {}

        for feature, weight in feature_weights:
            # Estrae il nome della feature dalla stringa dell'intervallo
            feature_name = feature.split("_")[0] if "_" in feature else feature.split(" ")[0]
            
            # Estrae i valori numerici dalla stringa della feature
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', feature)
            values = [float(num) for num in numbers]
            
            # Se ci sono valori numerici, li usa, altrimenti usa solo il peso
            if values:
                # Puoi scegliere quale valore usare (il primo, l'ultimo, la media, ecc.)
                value = values[-1]  # Uso l'ultimo valore come esempio
                importance_dict[feature_name] = (value, abs(weight))
            else:
                importance_dict[feature_name] = (None, abs(weight))
                
        # dictionary of feature importances for each instance
        all_importances[f"instance_{i}"] = importance_dict

        df = pd.DataFrame(all_importances)
        df.to_csv("feature_importances.csv")
        
    return all_importances

def lime_based_resampling(X, explanations, instance_to_oversampling, scale_factor=1.0):
    """
    Resampling based on LIME explanations.
    """    
    # Extract feature importances from LIME explanations
    feature_importances = extract_feature_importances(explanations)
        
    new_samples = []

    df = X.copy()
    df["opt_config"] = df["opt_config"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df = df["opt_config"].apply(pd.Series)
    
    for index in range(instance_to_oversampling):
        new_sample = df.iloc[index].copy()
        print(new_sample)
        # Using index since we have a dataseries, not a dataframe
        for feature in new_sample.index:
            # Usa la media originale e la varianza basata su LIME
            mean = new_sample[feature]
            variance = feature_importances[feature] * scale_factor
            
            # Limita la varianza minima per evitare campioni troppo simili
            variance = max(variance, 0.01)
            
            # Genera un nuovo valore dalla distribuzione normale
            new_value = np.random.normal(mean, np.sqrt(variance))
            
            # Aggiorna il valore della feature
            new_sample[feature] = new_value
        
        new_samples.append(new_sample)
    
    # Converti i campioni in DataFrame
    resampled_df = pd.DataFrame(new_samples, columns=X.columns)

    return resampled_df

def main():
    # Ottieni le spiegazioni LIME
    csv_path = "./Matching_and_verifier/verifier_results/invalid_conifigs/"
    model_path = "regressor_SCS.joblib"
    explanations = explain_prediction_with_lime(csv_path, model_path, num_features=20)
    X = pd.read_csv(csv_path)

    # Genera nuovi campioni
    resampled_X = lime_based_resampling(X, explanations, X.shape[0], scale_factor=0.1)
    
    # Salva i dati ricampionati
    output_path = "lime_resampled_data.csv"
    resampled_X.to_csv(output_path, index=False)
    print(f"Dati ricampionati salvati in: {output_path}")
    
if __name__ == "__main__":
    main()
