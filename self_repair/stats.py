from typing import List
import statistics
import pickle

class Stat:
    def __init__(self, method_name: str, epsilon_points: List[float], th_mission_satisfied: float = 0.5):
        self.method_name = method_name
        self.epsilon_points = epsilon_points
        self.th_mission_satisfied = th_mission_satisfied
        self.n_mission_success = 0
        self.n_mission_failed = 0
        self.__evaluate_missions()

    def __evaluate_missions(self):
        for epsilon in self.epsilon_points:
            if epsilon >= self.th_mission_satisfied:
                self.n_mission_success += 1
            else:
                self.n_mission_failed += 1

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

    def get_n_mission_success(self) -> int:
        return self.n_mission_success

    def get_n_mission_failed(self) -> int:
        return self.n_mission_failed

    def to_pickle(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)