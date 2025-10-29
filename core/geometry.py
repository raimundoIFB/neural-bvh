# core/geometry.py
import numpy as np

def random_triangles(n_tri=100, scale=1.0, seed=0):
    np.random.seed(seed)
    # cada triângulo: 3 vértices (x,y,z)
    tris = np.random.randn(n_tri, 3, 3) * scale
    return tris  # shape (n_tri, 3, 3)

def triangle_centroids(tris):
    return tris.mean(axis=1)  # (n_tri, 3)

def triangle_area(tri):
    # tri: (3,3)
    a = tri[1] - tri[0]
    b = tri[2] - tri[0]
    return 0.5 * np.linalg.norm(np.cross(a, b))

def triangles_surface_areas(tris):
    return np.array([triangle_area(t) for t in tris])
