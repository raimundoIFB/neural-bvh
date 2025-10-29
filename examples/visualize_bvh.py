# examples/visualize_bvh.py
import matplotlib.pyplot as plt
from core.geometry import random_triangles
from core.bvh import build_bvh, collect_leaves
from core.aabb import AABB

def plot_aabbs(node, ax, depth=0):
    if node is None: return
    a = node.aabb
    # project to XY plane
    x0, y0 = a.mins[0], a.mins[1]
    w = a.maxs[0] - a.mins[0]
    h = a.maxs[1] - a.mins[1]
    rect = plt.Rectangle((x0,y0), w, h, fill=False, edgecolor='C{}'.format(depth%10), linewidth=0.8)
    ax.add_patch(rect)
    if not node.is_leaf:
        plot_aabbs(node.left, ax, depth+1)
        plot_aabbs(node.right, ax, depth+1)

tris = random_triangles(200, scale=1.0, seed=1)
bvh = build_bvh(tris, max_leaf_size=8)
fig, ax = plt.subplots(figsize=(6,6))
plot_aabbs(bvh, ax)
ax.set_aspect('equal')
plt.title("Projeção XY das AABBs do BVH")
plt.show()
