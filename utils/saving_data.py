from typing import List, Dict
import os 
import pickle

def save_results(new_results: List[Dict], save_path: str) -> None:
    try:
        with open(save_path, "rb") as f:
            existing = pickle.load(f)
    except (FileNotFoundError, EOFError):
        existing = []

    combined = existing + new_results

    with open(save_path, "wb") as f:
        pickle.dump(combined, f)

def load_existing_results(save_path: str) -> List[Dict]:
    """Load existing results from a Pickle file."""
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)
    return []


