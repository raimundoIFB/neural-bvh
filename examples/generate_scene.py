# examples/generate_scene.py
import os
import numpy as np
from core.geometry import random_triangles
import trimesh

out_dir = "data/scenes"
os.makedirs(out_dir, exist_ok=True)

tris = random_triangles(200, scale=1.0, seed=42)
# criar um mesh trimesh apenas para visualização/export
vertices = tris.reshape(-1,3)
faces = np.arange(len(vertices)).reshape(-1,3)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export(os.path.join(out_dir, "scene_toy.obj"))
print("Cena salva em data/scenes/scene_toy.obj")
