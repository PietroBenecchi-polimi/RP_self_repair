import pandas as pd
from typing import List, Dict, Any
from itertools import zip_longest


# Transformation rules
transform_rules = {
    'HUM_1_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_1_AGE': {'y': 0, 'e': 1},
    'HUM_1_STA': {'s': 1, 'h': 0, 'u': 2},
    'HUM_2_FW': {'free': 2.0, 'foc': 0.0, 'distr': 1.0},
    'HUM_2_AGE': {'y': 0, 'e': 1},
    'HUM_2_STA': {'s': 1, 'h': 0, 'u': 2},
}

# Reverse rules
reverse_rules = {
    col: {v: k for k, v in mapping.items()} for col, mapping in transform_rules.items()
}

def get_transformation_rules():
    return transform_rules

def categorical_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for column, mapping in transform_rules.items():
        if column in df_copy.columns:
            df_copy[column] = df_copy[column].map(mapping)
    return df_copy

def numeric_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for column, mapping in reverse_rules.items():
        if column in df_copy.columns:
            df_copy[column] = df_copy[column].map(mapping)
    return df_copy

def load_dataset_for_regressor(data_path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    
    data = data.drop(["PRSCS_LB","PRSCS_UB","FTG_HUM_1_LB","FTG_HUM_1_UB",
                     "FTG_HUM_1","FTG_HUM_2_LB","FTG_HUM_2_UB","FTG_HUM_2"], axis =1)
    
    data = categorical_to_numeric(data)
    
    return data

def fromOptimizerToMC(data):
    # Convert categorical columns to numeric
    data = numeric_to_categorical(data)
    # Combine AGE and STAT into a new column
    data['AGE/STAT 1'] = data['HUM_1_AGE'].astype(str) + '/' + data['HUM_1_STA'].astype(str)
    data['AGE/STAT 2'] = data['HUM_2_AGE'].astype(str) + '/' + data['HUM_2_STA'].astype(str)

    data['Position 1'] = data['HUM_1_POS_X'].astype(str) + ', ' + data['HUM_1_POS_Y'].astype(str)
    data['Position 2'] = data['HUM_2_POS_X'].astype(str) + ', ' + data['HUM_2_POS_Y'].astype(str) 

    data.drop(columns=['HUM_1_POS_X', 'HUM_1_POS_Y', 'HUM_2_POS_X', 'HUM_2_POS_Y', 'HUM_1_AGE', 'HUM_2_AGE',
                      'HUM_1_STA', 'HUM_2_STA'], inplace=True) 
                      
    # Extract new columns
    age_stat_1 = data.pop('AGE/STAT 1')
    age_stat_2 = data.pop('AGE/STAT 2')
    vel_1 = data.pop('Position 1')
    vel_2 = data.pop('Position 2')

    # Desidered position
    data.insert(9, 'AGE/STAT 1', age_stat_1)
    data.insert(11, 'AGE/STAT 2', age_stat_2)
    data.insert(12, 'Position 1', vel_1)
    data.insert(13, 'Position 2', vel_2)
    
    data['PRSCS_LOWER_BOUND'] = 0
    data['PRSCS_UPPER_BOUND'] = 0
    data['FTG_HUM_1'] = 0
    data['FTG_HUM_2'] = 0

    return data

def fromMCtoOptimizer(data) -> pd.DataFrame:
    mask = ~((data['PRSCS_LOWER_BOUND'] == 0) & (data['PRSCS_UPPER_BOUND'] == 0) & (data['FTG_HUM_1'] == 0) & (data['FTG_HUM_2'] == 0))
    data = data[mask]

    data['SCS'] = (data['PRSCS_LOWER_BOUND'] + data['PRSCS_UPPER_BOUND']) / 2

    # Split AGE/STAT columns back into original components
    data['HUM_1_FTG'] = data['HUM_1_FTG'].astype(str)
    data['HUM_2_FTG'] = data['HUM_2_FTG'].astype(str)

    data[['HUM_1_AGE', 'HUM_1_STA']] = data['HUM_1_FTG'].str.split('/', expand=True)
    data[['HUM_2_AGE', 'HUM_2_STA']] = data['HUM_2_FTG'].str.split('/', expand=True)
    
    data = data.rename(columns={'PROGRESS': 'PRGS'})

    # Split Position columns back into original X and Y
    data['HUM_1_POS'] = data['HUM_1_POS'].astype(str)
    data['HUM_2_POS'] = data['HUM_2_POS'].astype(str)

    data[['HUM_1_POS_X', 'HUM_1_POS_Y']] = data['HUM_1_POS'].str.split(', ', expand=True).astype(float)
    data[['HUM_2_POS_X', 'HUM_2_POS_Y']] = data['HUM_2_POS'].str.split(', ', expand=True).astype(float)

    # drop combined columns
    data.drop(columns=['HUM_1_FTG', 'HUM_2_FTG', 'HUM_1_POS', 'HUM_2_POS'], inplace=True)

    # Remove columns scs and ftg (should be changed if test on fatigue)
    data.drop(columns=['PRSCS_LOWER_BOUND', 'PRSCS_UPPER_BOUND', 'FTG_HUM_1', 'FTG_HUM_2'], inplace=True)
    
    # Extract new columns
    age_1 = data.pop('HUM_1_AGE')
    age_2 = data.pop('HUM_2_AGE')
    stat_1 = data.pop('HUM_1_STA')
    stat_2 = data.pop('HUM_2_STA')
    vel_1 = data.pop('ROB_1_VEL')
    chg_1 = data.pop('ROB_1_CHG')

    # Desidered position
    data.insert(9, 'HUM_1_AGE', age_1)
    data.insert(11, 'HUM_2_AGE', age_2)
    data.insert(10, 'HUM_1_STA', stat_1)
    data.insert(13, 'HUM_2_STA', stat_2)
    data.insert(19, 'ROB_1_VEL', vel_1)
    data.insert(20, 'ROB_1_CHG', chg_1)
    
    data = categorical_to_numeric(data) 
    
    return data

def castIntegerFeatures(data):
    for feature in data.columns:
        if feature in ["PSCS__TAU", "PRGS"]:
            data[feature] = data[feature].astype(int)
    return data

def audit_stats(stats_per_points):
    print("== audit ==")
    for i, exp in enumerate(stats_per_points[:10]):  # sample first 2 experiments
        stats = exp.get('stats', [])
        print(f"exp {i}: type(stats)={type(stats)}, len={len(stats)}")
        if stats:
            s0 = stats[5]
            print("first stat type:", type(s0))
            print("has neighbours_optimized:", hasattr(s0, "neighbours_optimized"))
            print("has neighbours_validation:", hasattr(s0, "neighbours_validation"))
            try:
                print("repr:\n", repr(s0))
            except Exception as e:
                print("repr failed:", e)

def audit_df(df):
    print("DF columns:", list(df.columns))
    print(df.head(3))

def _extract_scs_from_neigh_list(neigh_list) -> list[float]:
    """Ritorna una lista piatta di SCS da una lista di float/DF/Series/dict."""
    out: list[float] = []
    if not isinstance(neigh_list, list):
        return out
    for item in neigh_list:
        # caso float/int
        if isinstance(item, (int, float)):
            out.append(float(item))
            continue

        # caso pandas DataFrame/Series
        if isinstance(item, pd.DataFrame):
            if 'SCS' in item.columns:
                out.extend(item['SCS'].astype(float).tolist())
            continue
        if isinstance(item, pd.Series):
            # se la series rappresenta già SCS, usala; altrimenti prova a convertire
            try:
                out.extend(pd.Series(item, dtype=float).tolist())
            except Exception:
                pass
            continue

        # caso dict con chiave 'SCS'
        if isinstance(item, dict) and 'SCS' in item:
            v = item['SCS']
            if isinstance(v, (list, tuple)):
                out.extend([float(x) for x in v])
            elif isinstance(v, pd.Series):
                out.extend(v.astype(float).tolist())
            elif isinstance(v, pd.DataFrame) and 'SCS' in v.columns:
                out.extend(v['SCS'].astype(float).tolist())
            elif isinstance(v, (int, float)):
                out.append(float(v))
            # altri tipi: ignora
    return out

def _normalize_epsilons(raw_eps) -> list[float]:
    """Converte epsilon_points a lista di float.
       Se è un DataFrame con 'SCS', usa quella colonna."""
    if raw_eps is None:
        return []
    if isinstance(raw_eps, pd.DataFrame):
        if 'SCS' in raw_eps.columns:
            return raw_eps['SCS'].astype(float).tolist()
        return []
    if isinstance(raw_eps, pd.Series):
        try:
            return raw_eps.astype(float).tolist()
        except Exception:
            return []
    if isinstance(raw_eps, (list, tuple)):
        return [float(x) for x in raw_eps]
    if isinstance(raw_eps, (int, float)):
        return [float(raw_eps)]
    # fallback: prova a iterare
    try:
        return [float(x) for x in list(raw_eps)]
    except Exception:
        return []

def process_results(stats_per_points: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Crea un DataFrame flatten:
      - Epsilons: da list/scalar o da DataFrame['SCS'] quando opportuno.
      - neighbours_optimized / neighbours_validation: estrae TUTTE le SCS
        (float diretti o tutte le righe della colonna SCS dei DataFrame).
      - Allinea lunghezze via zip_longest; se c'è un solo epsilon e molte SCS, fa broadcast.
    """
    data = []

    for experiment in stats_per_points:
        r_points = experiment['regressor_points']
        s_points = experiment['resampling_points']
        stats_list = experiment['stats']  # list[Stat]

        for stat in stats_list:
            method = getattr(stat, "method_name", None)

            # 1) Epsilons
            raw_eps = getattr(stat, "epsilon_points", None)
            eps_list = _normalize_epsilons(raw_eps)

            # 2) Neighbours → estrai TUTTE le SCS presenti
            opt_vals = _extract_scs_from_neigh_list(getattr(stat, "neighbours_optimized", []))
            val_vals = _extract_scs_from_neigh_list(getattr(stat, "neighbours_validation", []))

            # 3) Allineamento lunghezze
            n = max(len(eps_list), len(opt_vals), len(val_vals), 0)
            if len(eps_list) == 1 and n > 1:
                eps_list = [eps_list[0]] * n
            # pad neighbours
            if len(opt_vals) < n:
                opt_vals = opt_vals + [None] * (n - len(opt_vals))
            if len(val_vals) < n:
                val_vals = val_vals + [None] * (n - len(val_vals))

            # 4) Costruisci righe (se eps mancante, lo lasciamo None: gestibile a valle)
            for eps, opt, val in zip_longest(eps_list, opt_vals, val_vals, fillvalue=None):
                data.append({
                    'Method': method,
                    'Regressor Points': r_points,
                    'Resampling Points': s_points,
                    'Epsilons': eps,
                    'neighbours_optimized': opt,
                    'neighbours_validation': val,
                })

    df = pd.DataFrame(data)

    # garantisci colonne
    for col in ['neighbours_optimized', 'neighbours_validation', 'Epsilons']:
        if col not in df.columns:
            df[col] = None

    return df
