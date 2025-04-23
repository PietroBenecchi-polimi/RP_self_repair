import math
from typing import List


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return '{:.2f}, {:.2f}'.format(self.x, self.y)

    def __hash__(self):
        return hash(str(self))

    @staticmethod
    def parse(s: str):
        return Point(float(s.split(',')[0]), float(s.split(',')[1]))

    def dist_from(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Area:
    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point):
        self.corners = [p1, p2, p3, p4]

    def contains_point(self, p: Point):
        return self.corners[0].x <= p.x <= self.corners[2].x and self.corners[0].y <= p.y <= self.corners[1].y


class Layout:
    def __init__(self, areas: List[Area], inter_pts: List[Point], max_neigh: int):
        self.areas = areas
        self.inter_pts = inter_pts
        self.max_neigh = max_neigh
