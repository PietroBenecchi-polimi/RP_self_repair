from typing import List
import statistics
import pickle

class Stat:
    def __init__(self, method_name: str, epsilon_points: List[float], neighbours_optimized: List[float], neighbours_validation: List[float]):
        self.method_name = method_name
        self.epsilon_points = epsilon_points
        self.neighbours_optimized = neighbours_optimized
        self.neighbours_validation = neighbours_validation


    def __repr__(self):
        return (
            f"Stat(\n"
            f"  method_name='{self.method_name}',\n"
            f"  epsilon_points={self.epsilon_points},\n"
            f"  neighbours_optimized={self.neighbours_optimized},\n"
            f"  neighbours_validation={self.neighbours_validation}\n"
            f")"
        )

    def get_method_name(self) -> str:
        return self.method_name

    def get_epsilon_points(self) -> List[float]:
        return self.epsilon_points

    def get_average_epsilon(self) -> float:
        if not self.epsilon_points:
            return 0.0
        return sum(self.epsilon_points) / len(self.epsilon_points)

    def get_median_epsilon(self) -> float:
        if not self.epsilon_points:
            return 0.0
        return statistics.median(self.epsilon_points)

    def to_pickle(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)