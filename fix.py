from typing import List, Dict, Any
import pandas as pd
import pickle
from self_repair.stats import Stat
import numpy as np

def process_results(stats_per_points: List[Dict]) -> pd.DataFrame:
    """Flatten the list of stats into a DataFrame suitable for boxplotting."""
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']
        stats: list[Stat] = experiment['stats']

        for stat in stats:
            method = stat.get_method_name()
            print(f"Processing method: {method}.")
            epsilon_points = stat.get_epsilon_points()

            opt_vals = extract_scs_values(getattr(stat, "neighbours_optimized", []))
            val_vals = extract_scs_values(getattr(stat, "neighbours_validation", []))

            if(isinstance(epsilon_points, list)):
                for opt_s, val_s in zip(opt_vals, val_vals):
                    data.append({
                        'Method': method,
                        'Regressor Points': r_points,
                        'Resampling Points': s_points,
                        'Neighbours_optimized': opt_s, 
                        'Neighbours_validation': val_s 
                    })
            elif(isinstance(epsilon_points, Dict)):
                for opt_s, val_s in zip(opt_vals, val_vals):
                    data.append({
                        'Method': method,
                        'Regressor Points': r_points,
                        'Resampling Points': s_points,
                        'Neighbours_optimized': opt_s, 
                        'Neighbours_validation': val_s 
                    })

    return pd.DataFrame(data)

def extract_scs_values(opt_vals) -> list[float]:
    """
    Estrae una lista di float SCS da:
    - DataFrame con colonna 'SCS'
    - Series (numerica o oggetti)
    - liste/tuple (numeriche o di dict con chiave 'SCS')
    - singolo valore numerico
    """
    # --- DataFrame ---
    if isinstance(opt_vals, pd.DataFrame):
        if 'SCS' in opt_vals.columns:
            return opt_vals['SCS'].astype(float).tolist()
        return []

    # --- Series ---
    if isinstance(opt_vals, pd.Series):
        s = opt_vals.dropna()
        if s.empty:
            return []
        # Caso Series numerica: già i valori SCS
        if np.issubdtype(s.dtype, np.number):
            return s.astype(float).tolist()
        # Caso Series di oggetti: guarda il primo elemento non nullo
        first = s.iloc[0]
        # dict con chiave 'SCS'
        if isinstance(first, dict) and 'SCS' in first:
            return [float(x['SCS']) for x in s if isinstance(x, dict) and 'SCS' in x]
        # Series annidate con indice 'SCS'
        if isinstance(first, pd.Series) and 'SCS' in first.index:
            return [float(x['SCS']) for x in s if isinstance(x, pd.Series) and 'SCS' in x.index]
        # stringhe numeriche
        try:
            return s.astype(float).tolist()
        except Exception:
            return []

    # --- lista/tupla ---
    if isinstance(opt_vals, (list, tuple)):
        if not opt_vals:
            return []
        # lista numerica
        if all(isinstance(x, (int, float, np.floating, np.integer)) or (isinstance(x, str) and x.replace('.','',1).isdigit())
               for x in opt_vals if x is not None):
            return [float(x) for x in opt_vals if x is not None]
        # lista di dict con 'SCS'
        if all(isinstance(x, dict) and 'SCS' in x for x in opt_vals if x is not None):
            return [float(x['SCS']) for x in opt_vals if x is not None]
        return []

    # --- singolo valore ---
    try:
        return [float(opt_vals)]
    except Exception:
        return []

if __name__ == "__main__":
    pkl_path = 'visualization/data/data_double_regressors_pietro/oversampling_results_invalid_configs.pkl'
    with open(pkl_path, "rb") as f:
        stats_per_points = pickle.load(f)
    
    df = process_results(stats_per_points)
    df.to_csv('processed_oversampling_results_invalid_configs.csv', index=False)
