# core/aabb.py
import numpy as np

class AABB:
    def __init__(self, mins=None, maxs=None):
        if mins is None:
            self.mins = np.array([np.inf, np.inf, np.inf], dtype=float)
            self.maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        else:
            self.mins = np.array(mins, dtype=float)
            self.maxs = np.array(maxs, dtype=float)

    def expand_by_points(self, pts):
        self.mins = np.minimum(self.mins, np.min(pts, axis=0))
        self.maxs = np.maximum(self.maxs, np.max(pts, axis=0))

    def union(self, other):
        return AABB(np.minimum(self.mins, other.mins), np.maximum(self.maxs, other.maxs))

    def surface_area(self):
        d = np.maximum(self.maxs - self.mins, 0.0)
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

    def center(self):
        return 0.5 * (self.mins + self.maxs)
