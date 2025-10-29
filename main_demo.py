    # main_demo.py
from examples.generate_scene import *
from neural.train import run_training
from neural.evaluate import compare
from core.geometry import random_triangles
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Gerando cena de teste...")
#    tris = random_triangles(500, scale=1.2, seed=123)
    tris = random_triangles(2000, scale=0.2)
    print("Treinando modelo (rápido toy)...")
    model = run_training(epochs=6)
    print("Comparando BVH tradicional vs BVH neural (EPO)...")
    res = compare(tris, model)
    print("Resultado EPO:", res)
