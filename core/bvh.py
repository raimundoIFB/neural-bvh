# core/bvh.py
import numpy as np
from .aabb import AABB
from .metrics import compute_sah_for_split

class BVHNode:
    def __init__(self, aabb=None, tri_indices=None, left=None, right=None):
        self.aabb = aabb
        self.tri_indices = tri_indices if tri_indices is not None else []
        self.left = left
        self.right = right
        self.is_leaf = (left is None and right is None)

def build_bvh(tris, tri_indices=None, max_leaf_size=8, depth=0):
    if tri_indices is None:
        tri_indices = list(range(len(tris)))
    node_aabb = AABB()
    for i in tri_indices:
        node_aabb.expand_by_points(tris[i])
    # leaf
    if len(tri_indices) <= max_leaf_size:
        return BVHNode(aabb=node_aabb, tri_indices=tri_indices)
    # try splits along axes using centroid median and SAH evaluation
    cents = np.array([tris[i].mean(axis=0) for i in tri_indices])
    best_cost = np.inf
    best = None
    for axis in range(3):
        # candidate split positions: quantiles of centroids
        candidates = np.unique(np.percentile(cents[:,axis], [10,30,50,70,90]))
        for pos in candidates:
            cost, left_idx, right_idx = compute_sah_for_split(tri_indices, tris, axis, pos)
            if len(left_idx)==0 or len(right_idx)==0:
                continue
            if cost < best_cost:
                best_cost = cost
                best = (left_idx, right_idx)
    if best is None:
        # fall back to median split
        axis = np.argmax(node_aabb.maxs - node_aabb.mins)
        order = np.argsort(cents[:,axis])
        mid = len(order)//2
        left_idx = [tri_indices[i] for i in order[:mid]]
        right_idx = [tri_indices[i] for i in order[mid:]]
    else:
        left_idx, right_idx = best
    left = build_bvh(tris, left_idx, max_leaf_size, depth+1)
    right = build_bvh(tris, right_idx, max_leaf_size, depth+1)
    node = BVHNode(aabb=node_aabb, left=left, right=right)
    node.is_leaf = False
    return node

def collect_leaves(node):
    """
    Coleta folhas de uma BVH. Funciona tanto com objetos (classe Node)
    quanto com dicionários (usados pelo BVH neural).
    """
    if node is None:
        return []

    # Caso: estrutura dicionário (BVH neural)
    if isinstance(node, dict):
        if "leaf" in node:
            return [node["leaf"]]
        leaves = []
        leaves.extend(collect_leaves(node.get("left")))
        leaves.extend(collect_leaves(node.get("right")))
        return leaves

    # Caso: estrutura de classe tradicional
    if getattr(node, "is_leaf", False):
        return [node.triangles if hasattr(node, "triangles") else []]

    leaves = []
    leaves.extend(collect_leaves(getattr(node, "left", None)))
    leaves.extend(collect_leaves(getattr(node, "right", None)))
    return leaves
