# core/metrics.py
import numpy as np
from .aabb import AABB

def sah_cost(node_aabb, left_aabb, right_aabb, n_left, n_right, traversal_cost=1.0, intersection_cost=1.0):
    # SAH = traversal_cost + (area(left)/area(node))*n_left*intersection_cost + same for right
    area_node = node_aabb.surface_area()
    if area_node == 0:
        return np.inf
    cost = traversal_cost
    cost += (left_aabb.surface_area() / area_node) * n_left * intersection_cost
    cost += (right_aabb.surface_area() / area_node) * n_right * intersection_cost
    return cost

def compute_sah_for_split(tri_indices, tris, axis, split_pos):
    # split tri_indices by centroid along axis
    cents = np.array([tris[i].mean(axis=0) for i in tri_indices])
    left_idx = [tri_indices[i] for i in range(len(tri_indices)) if cents[i,axis] <= split_pos]
    right_idx = [tri_indices[i] for i in range(len(tri_indices)) if cents[i,axis] > split_pos]
    # compute AABBs
    left_aabb = AABB(); right_aabb = AABB(); node_aabb = AABB()
    for i in tri_indices:
        node_aabb.expand_by_points(tris[i])
    for i in left_idx:
        left_aabb.expand_by_points(tris[i])
    for i in right_idx:
        right_aabb.expand_by_points(tris[i])
    return sah_cost(node_aabb, left_aabb, right_aabb, len(left_idx), len(right_idx)), left_idx, right_idx

def compute_epo(tris, leaves):
    """
    Calcula a métrica Expected Primitive Overlap (EPO)
    para um conjunto de folhas de uma BVH.

    - tris: numpy array (N, 3, 3) com triângulos.
    - leaves: lista de listas de índices de triângulos.
    """
    total = 0
    multi = 0
    counted = set()

    for leaf in leaves:
        total += len(leaf)
        for i in leaf:
            if i in counted:
                multi += 1
            counted.add(i)

    frac = multi / total if total > 0 else 0.0
    return {"multi": multi, "total": total, "frac": frac}
