from typing import List, Dict
import os 
import pickle

def save_results(stats_per_points: List[Dict], save_path: str) -> None:
    """Save results to a Pickle file."""
    with open(save_path, "wb") as f:
        pickle.dump(stats_per_points, f)

def load_existing_results(save_path: str) -> List[Dict]:
    """Load existing results from a Pickle file."""
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)
    return []


